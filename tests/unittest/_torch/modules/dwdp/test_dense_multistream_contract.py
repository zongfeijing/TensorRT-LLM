# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The multi-stream contract of dense DWDP on Kimi-K3, checked on the source.

``KimiLinearDecoderLayer`` declares ``DENSE_DWDP_MULTI_STREAM_SAFE`` on an audit of the forks onto
auxiliary streams in the code it runs, so dense DWDP runs it with multi-stream execution (the
contract is ``_CONTRACT``).  A fork that joins too late lets the next copy into a slot overwrite the
weights under the read: wrong tokens, and no error.  So the ``maybe_execute_in_parallel``,
``torch.cuda.stream`` and ``wait_stream`` call sites of those modules, counted per enclosing
function, must be the audited ones.  A static source scan: no GPU, no model.
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

import tensorrt_llm

pytestmark = pytest.mark.cpu_only

_TORCH_DIR = Path(tensorrt_llm.__file__).parent / "_torch"

# What a Kimi-K3 decoder layer runs between H1 and H3: the layer with its MoE block, the KDA mixer,
# and the K3 MLA with its MLA base class.  The routed experts read no dense weight, and the
# attention backends read only activations and the KV cache.
_K3_MODULES = (
    "models/modeling_kimi_linear.py",
    "modules/kimi_kda",
    "modules/kimi_k3_mla",
    "attention/mla.py",
)

# The audited fork points: module -> enclosing function -> calls.  The docstring of
# ``KimiLinearDecoderLayer.DENSE_DWDP_MULTI_STREAM_SAFE`` describes them.
_AUDITED_FORKS = {
    "models/modeling_kimi_linear.py": {
        # The shared expert, joined before the MoE block returns, and the latent down projection
        # (CUTEDSL and CUTEDSL_FC12 only), which the routed experts join.
        "KimiK3MoERuntime.forward": ("maybe_execute_in_parallel", "torch.cuda.stream"),
    },
    "modules/kimi_kda/kimi_kda_mixer.py": {
        # The [f_a | b] projection chain for at most 128 rows, joined inside the mixer.
        "KimiKDALinearAttention.forward_decode": ("maybe_execute_in_parallel",),
        "KimiKDALinearAttention._project_verify_inputs": ("maybe_execute_in_parallel",),
    },
    "attention/mla.py": {
        # The latent layernorm and the latent-cache concat, joined inside the attention call.
        "MLA._forward_impl": ("maybe_execute_in_parallel",) * 2,
        # The generation RoPE beside the BF16 or the FP8 absorption BMM, joined there too.
        "MLA.forward_absorption_generation": ("maybe_execute_in_parallel",) * 2,
    },
}

_CONTRACT = (
    "KimiLinearDecoderLayer declares DENSE_DWDP_MULTI_STREAM_SAFE, so dense DWDP runs it with "
    "multi-stream execution: every read of a covered weight on an auxiliary stream must fork after "
    "H1 (wait_layer) and join the compute stream before H2 (attention_done) for attention weights "
    "or H3 (layer_done) for MoE-dense weights. Audit the fork point against that, then list it in "
    "_AUDITED_FORKS and in the docstring of DENSE_DWDP_MULTI_STREAM_SAFE."
)


def _fork_call(func: ast.expr) -> str | None:
    """The fork call that ``func``, the callee of a call, names; None for any other call."""
    names = []
    while isinstance(func, ast.Attribute):
        names.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        names.append(func.id)
    if names and names[0] in ("maybe_execute_in_parallel", "wait_stream"):
        return names[0]
    return "torch.cuda.stream" if names == ["stream", "cuda", "torch"] else None


def _fork_sites(path: str, tree: ast.Module) -> list[tuple[str, str, str]]:
    """``(path, enclosing function, call)`` of every fork call in ``tree``."""
    sites = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                child_scope = f"{scope}.{child.name}" if scope else child.name
            elif isinstance(child, ast.Call) and (call := _fork_call(child.func)):
                sites.append((path, scope, call))
            visit(child, child_scope)

    visit(tree, "")
    return sites


def _k3_module_trees() -> dict[str, ast.Module]:
    files = []
    for entry in _K3_MODULES:
        path = _TORCH_DIR / entry
        files += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    return {
        str(file.relative_to(_TORCH_DIR)): ast.parse(file.read_text(), filename=str(file))
        for file in files
    }


def _unaudited_and_missing(trees: dict[str, ast.Module]) -> tuple[list, list]:
    found = Counter(site for path, tree in trees.items() for site in _fork_sites(path, tree))
    audited = Counter(
        (path, scope, call)
        for path, scopes in _AUDITED_FORKS.items()
        for scope, calls in scopes.items()
        for call in calls
    )
    return sorted((found - audited).elements()), sorted((audited - found).elements())


def test_k3_fork_points_are_the_audited_ones():
    unaudited, missing = _unaudited_and_missing(_k3_module_trees())
    assert not unaudited, f"fork points outside the dense DWDP audit: {unaudited}. {_CONTRACT}"
    assert not missing, f"audited fork points gone from the code: {missing}; update _AUDITED_FORKS"


@pytest.mark.parametrize(
    "statement, call",
    [
        ("maybe_execute_in_parallel(f0, f1, e0, e1, aux)", "maybe_execute_in_parallel"),
        ("with torch.cuda.stream(aux): pass", "torch.cuda.stream"),
        ("torch.cuda.current_stream().wait_stream(aux)", "wait_stream"),
    ],
)
def test_the_check_catches_a_planted_fork(statement, call):
    # Planted beside the two audited forks of the MoE block: the sites of a function are counted.
    path, scope = "models/modeling_kimi_linear.py", "KimiK3MoERuntime.forward"
    trees = _k3_module_trees()
    function = trees[path]
    for name in scope.split("."):
        function = next(node for node in function.body if getattr(node, "name", None) == name)
    function.body.insert(0, ast.parse(statement).body[0])
    assert _unaudited_and_missing(trees) == ([(path, scope, call)], [])
