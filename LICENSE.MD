<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Activating a DeepSparse Commercial License
**TODO** add information about deepsparse-ent
A commercial license for the DeepSparse engine can be activated using the
`deepsparse.license` tool.

**TODO** add information about what the license does

To activate a license, a JSON Web Token (JWT) key is required. This can be provided
as a raw input to the `deepsparse.license` CLI or through a text file containing
only the token.

**TODO** add information on how to obtain a license

## Installation
The license tool comes installed as part of the deepsparse python package which
is installed through `pip`:

```bash
pip3 install deepsparse-ent
```


## Use
After pip installing deepsparse, the `license` script will be available through
the CLI.  The CLI takes one arguement which is either the raw JWT key or a path
to a text file containing only the key.

```bash
$ deepsparse.license <KEY-OR-PATH-TO-KEY>
```

If successfully validated, a response similar to the one below will be displayed,
and commercial use of deepsparse is valid for the duration of the license key.

```bash
**TODO** add example sucess splash message
```

If there is an error on validation
**TODO** add information on what do to with an error


For more information and questions about comercial use **TODO**
