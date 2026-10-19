# Third-party notices

This project is licensed under the Apache License, Version 2.0 (see `LICENSE`). The files below contain
text, prompts, grammar, or code derived from other projects. Each entry names the upstream project, its
license, and where the material appears in this repository.

| Upstream project | License | Where it appears | What it is |
|---|---|---|---|
| verl (volcengine/verl) | Apache-2.0 | `lh_platform/lh_platform/studies/execution.py` | Source fragments used as patch anchors for the upstream trainer file. |
| verl-agent (langfengQ/verl-agent) | Apache-2.0 | `lh_platform/lh_platform/training/g2po/actor_protocol.py` | ALFWorld prompt templates. About 60% of the template lines match the upstream prompt file line for line. |
| ToolBench (OpenBMB/ToolBench) | Apache-2.0 | `lh_platform/lh_platform/training/toolbench_actor_protocol.py` | The Call/Finish action grammar and return types. |
| flash-attention (Dao-AILab/flash-attention) | BSD-3-Clause | `lh_platform/lh_platform/studies/graphgpo/runner.py` | Reimplemented `index_first_axis`, `unpad_input` and `pad_input` helpers. |
| ACV (microsoft/ACV) | MIT | `lh_platform/lh_platform/test_time_adaptation/dover_official_intervention_public.py` | One short system-prompt sentence for the DoVer intervention step. |
| ALFWorld, ScienceWorld, Terminal-Bench, SWE-bench, GraphGPO, TraceElephant, Who&When | various | adapter modules under `benchmarks/`, `tracing/integrations/`, `studies/` | Names, repository paths and short quoted lines used for verification and anchors. No upstream code is vendored; the adapters load upstream checkouts that the user supplies. |

Upstream projects are not included in this repository.

## Full text: flash-attention (BSD-3-Clause)

BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
