# Software Inventory

This document lists all third-party software packages used in this project, including their versions, licenses, authors, and sources.

**Generated:** Automatically from dependency files
**Last Updated:** 2025-12-13
**Generation Script:** `scripts/tools/generate_software_inventory.py`

## How to Regenerate

To regenerate this inventory with the latest package information:

```bash
# Activate virtual environment
source env/bin/activate

# Run the generation script
python scripts/tools/generate_software_inventory.py
```

The script automatically:
- Parses `requirements.txt`, `requirements.docker.txt`, and `scripts/requirements_synthetic_data.txt`
- Parses `pyproject.toml` for Python dependencies and dev dependencies
- Parses root `package.json` for Node.js dev dependencies (tooling)
- Parses `src/ui/web/package.json` for frontend dependencies (React, Material-UI, etc.)
- Queries PyPI and npm registries for package metadata
- Removes duplicates and formats the data into this table

## Python Packages (PyPI)

| Package Name | Version | License | License URL | Author | Source | Distribution Method |
|--------------|---------|---------|-------------|--------|--------|---------------------|
| aiohttp | 3.8.0 | Apache 2 | https://github.com/aio-libs/aiohttp | N/A | PyPI | pip |
| asyncpg | 0.29.0 | Apache License, Version 2.0 | https://pypi.org/project/asyncpg/ | MagicStack Inc <hello@magic.io> | PyPI | pip |
| bacpypes3 | 0.0.0 | N/A | https://pypi.org/project/bacpypes3/ | N/A | PyPI | pip |
| bcrypt | 4.0.0 | Apache License, Version 2.0 | https://github.com/pyca/bcrypt/ | The Python Cryptographic Authority developers <cryptography-dev@python.org> | PyPI | pip |
| click | 8.0.0 | BSD-3-Clause | https://palletsprojects.com/p/click/ | Armin Ronacher <armin.ronacher@active-4.com> | PyPI | pip |
| email-validator | 2.0.0 | CC0 (copyright waived) | https://github.com/JoshData/python-email-validator | Joshua Tauberer <jt@occams.info> | PyPI | pip |
| Faker | 19.0.0 | MIT License | https://github.com/joke2k/faker | joke2k <joke2k@gmail.com> | PyPI | pip |
| fastapi | 0.120.0 | MIT License | https://pypi.org/project/fastapi/ | Sebastián Ramírez <tiangolo@gmail.com> | PyPI | pip |
| httpx | 0.27.0 | BSD License | https://pypi.org/project/httpx/ | Tom Christie <tom@tomchristie.com> | PyPI | pip |
| langchain-core | 0.3.80 | MIT | https://pypi.org/project/langchain-core/ | N/A | PyPI | pip |
| langgraph | 0.2.30 | MIT | https://www.github.com/langchain-ai/langgraph | N/A | PyPI | pip |
| loguru | 0.7.0 | MIT license | https://github.com/Delgan/loguru | Delgan <delgan.py@gmail.com> | PyPI | pip |
| nemoguardrails | 0.19.0 | LICENSE.md | https://pypi.org/project/nemoguardrails/ | NVIDIA <nemoguardrails@nvidia.com> | PyPI | pip |
| numpy | 1.24.0 | BSD-3-Clause | https://www.numpy.org | Travis E. Oliphant et al. | PyPI | pip |
| paho-mqtt | 1.6.0 | Eclipse Public License v2.0 / Eclipse Distribution License v1.0 | http://eclipse.org/paho | Roger Light <roger@atchoo.org> | PyPI | pip |
| pandas | 1.2.4 | BSD | https://pandas.pydata.org | N/A | PyPI | pip |
| passlib | 1.7.4 | BSD | https://passlib.readthedocs.io | Eli Collins <elic@assurancetechnologies.com> | PyPI | pip |
| pillow | 10.3.0 | HPND | https://pypi.org/project/Pillow/ | "Jeffrey A. Clark" <aclark@aclark.net> | PyPI | pip |
| prometheus-client | 0.19.0 | Apache Software License 2.0 | https://github.com/prometheus/client_python | Brian Brazil <brian.brazil@robustperception.io> | PyPI | pip |
| psutil | 5.9.0 | BSD | https://github.com/giampaolo/psutil | Giampaolo Rodola <g.rodola@gmail.com> | PyPI | pip |
| psycopg | 3.0 | GNU Lesser General Public License v3 (LGPLv3) | https://psycopg.org/psycopg3/ | Daniele Varrazzo <daniele.varrazzo@gmail.com> | PyPI | pip |
| pydantic | 2.7.0 | MIT License | https://pypi.org/project/pydantic/ | Samuel Colvin <s@muelcolvin.com>, Eric Jolibois <em.jolibois@gmail.com>, Hasan Ramezani <hasan.r67@gmail.com>, Adrian Garcia Badaracco <1755071+adr... | PyPI | pip |
| PyJWT | 2.8.0 | MIT | https://github.com/jpadilla/pyjwt | Jose Padilla <hello@jpadilla.com> | PyPI | pip |
| pymilvus | 2.3.0 | Apache Software License | https://pypi.org/project/pymilvus/ | Milvus Team <milvus-team@zilliz.com> | PyPI | pip |
| pymodbus | 3.0.0 | BSD-3-Clause | https://github.com/riptideio/pymodbus/ | attr: pymodbus.__author__ | PyPI | pip |
| PyMuPDF | 1.23.0 | GNU AFFERO GPL 3.0 | https://pypi.org/project/PyMuPDF/ | Artifex <support@artifex.com> | PyPI | pip |
| pyserial | 3.5 | BSD | https://github.com/pyserial/pyserial | Chris Liechti <cliechti@gmx.net> | PyPI | pip |
| python-dotenv | 1.0.0 | BSD-3-Clause | https://github.com/theskumar/python-dotenv | Saurabh Kumar <me+github@saurabh-kumar.com> | PyPI | pip |
| python-multipart | 0.0.20 | Apache Software License | https://pypi.org/project/python-multipart/ | Andrew Dunham <andrew@du.nham.ca>, Marcelo Trylesinski <marcelotryle@gmail.com> | PyPI | pip |
| PyYAML | 6.0 | MIT | https://pyyaml.org/ | Kirill Simonov <xi@resolvent.net> | PyPI | pip |
| redis | 5.0.0 | MIT | https://github.com/redis/redis-py | Redis Inc. <oss@redis.com> | PyPI | pip |
| requests | 2.32.4 | Apache-2.0 | https://requests.readthedocs.io | Kenneth Reitz <me@kennethreitz.org> | PyPI | pip |
| scikit-learn | 1.5.0 | new BSD | https://scikit-learn.org | N/A | PyPI | pip |
| starlette | 0.49.1 | N/A | https://pypi.org/project/starlette/ | Tom Christie <tom@tomchristie.com> | PyPI | pip |
| tiktoken | 0.12.0 | MIT License | https://pypi.org/project/tiktoken/ | Shantanu Jain <shantanu@openai.com> | PyPI | pip |
| uvicorn | 0.30.1 | BSD License | https://pypi.org/project/uvicorn/ | Tom Christie <tom@tomchristie.com> | PyPI | pip |
| websockets | 11.0 | BSD-3-Clause | https://pypi.org/project/websockets/ | Aymeric Augustin <aymeric.augustin@m4x.org> | PyPI | pip |
| xgboost | 1.6.0 | Apache-2.0 | https://github.com/dmlc/xgboost | N/A | PyPI | pip |

## Node.js Packages (npm)

| Package Name | Version | License | License URL | Author | Source | Distribution Method |
|--------------|---------|---------|-------------|--------|--------|---------------------|
| @commitlint/cli | 19.8.1 | MIT | https://github.com/conventional-changelog/commitlint/blob/main/LICENSE | Mario Nebl <hello@herebecode.com> | npm | npm |
| @commitlint/config-conventional | 19.8.1 | MIT | https://github.com/conventional-changelog/commitlint/blob/main/LICENSE | Mario Nebl <hello@herebecode.com> | npm | npm |
| @craco/craco | 7.1.0 | Apache-2.0 | https://github.com/dilanx/craco/blob/main/LICENSE | Dilan Nair | npm | npm |
| @emotion/react | 11.10.0 | MIT | https://github.com/emotion-js/emotion.git#main/blob/main/LICENSE | Emotion Contributors | npm | npm |
| @emotion/styled | 11.10.0 | MIT | https://github.com/emotion-js/emotion.git#main/blob/main/LICENSE | N/A | npm | npm |
| @mui/icons-material | 5.10.0 | N/A | https://mui.com/material-ui/material-icons/ | N/A | npm | npm |
| @mui/material | 5.10.0 | MIT | https://github.com/mui/material-ui/blob/main/LICENSE | MUI Team | npm | npm |
| @mui/x-data-grid | 5.17.0 | MIT | https://github.com/mui/mui-x/blob/main/LICENSE | MUI Team | npm | npm |
| @semantic-release/changelog | 6.0.3 | MIT | https://github.com/semantic-release/changelog/blob/main/LICENSE | Pierre Vanduynslager | npm | npm |
| @semantic-release/exec | 7.1.0 | MIT | https://github.com/semantic-release/exec/blob/main/LICENSE | Pierre Vanduynslager | npm | npm |
| @semantic-release/git | 10.0.1 | MIT | https://github.com/semantic-release/git/blob/main/LICENSE | Pierre Vanduynslager | npm | npm |
| @semantic-release/github | 11.0.6 | MIT | https://github.com/semantic-release/github/blob/main/LICENSE | Pierre Vanduynslager | npm | npm |
| @testing-library/jest-dom | 5.16.4 | MIT | https://github.com/testing-library/jest-dom/blob/main/LICENSE | Ernesto Garcia <gnapse@gmail.com> | npm | npm |
| @testing-library/react | 13.3.0 | MIT | https://github.com/testing-library/react-testing-library/blob/main/LICENSE | Kent C. Dodds <me@kentcdodds.com> | npm | npm |
| @testing-library/user-event | 13.5.0 | MIT | https://github.com/testing-library/user-event/blob/main/LICENSE | Giorgio Polvara <polvara@gmail.com> | npm | npm |
| @types/jest | 27.5.2 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped/blob/main/LICENSE | N/A | npm | npm |
| @types/node | 16.11.56 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped/blob/main/LICENSE | N/A | npm | npm |
| @types/papaparse | 5.5.1 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped/blob/main/LICENSE | N/A | npm | npm |
| @types/react | 18.3.27 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped/blob/main/LICENSE | N/A | npm | npm |
| @types/react-copy-to-clipboard | 5.0.7 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped/blob/main/LICENSE | N/A | npm | npm |
| @types/react-dom | 18.3.7 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped/blob/main/LICENSE | N/A | npm | npm |
| @uiw/react-json-view | 2.0.0-alpha.39 | MIT | https://github.com/uiwjs/react-json-view/blob/main/LICENSE | Kenny Wang <wowohoo@qq.com> | npm | npm |
| axios | 1.8.3 | MIT | https://github.com/axios/axios/blob/main/LICENSE | Matt Zabriskie | npm | npm |
| commitizen | 4.3.1 | MIT | https://github.com/commitizen/cz-cli/blob/main/LICENSE | Jim Cummins <jimthedev@gmail.com> | npm | npm |
| conventional-changelog-conventionalcommits | 9.1.0 | ISC | https://github.com/conventional-changelog/conventional-changelog/blob/main/LICENSE | Ben Coe | npm | npm |
| cz-conventional-changelog | 3.3.0 | MIT | https://github.com/commitizen/cz-conventional-changelog/blob/main/LICENSE | Jim Cummins <jimthedev@gmail.com> | npm | npm |
| date-fns | 2.29.0 | MIT | https://github.com/date-fns/date-fns/blob/main/LICENSE | N/A | npm | npm |
| http-proxy-middleware | 3.0.5 | MIT | https://github.com/chimurai/http-proxy-middleware/blob/main/LICENSE | Steven Chim | npm | npm |
| husky | 9.1.7 | MIT | https://github.com/typicode/husky/blob/main/LICENSE | typicode | npm | npm |
| papaparse | 5.5.3 | MIT | https://github.com/mholt/PapaParse/blob/main/LICENSE | Matthew Holt | npm | npm |
| react | 18.2.0 | MIT | https://github.com/facebook/react/blob/main/LICENSE | N/A | npm | npm |
| react-copy-to-clipboard | 5.1.0 | MIT | https://github.com/nkbt/react-copy-to-clipboard/blob/main/LICENSE | Nik Butenko <nik@butenko.me> | npm | npm |
| react-dom | 18.2.0 | MIT | https://github.com/facebook/react/blob/main/LICENSE | N/A | npm | npm |
| react-query | 3.39.0 | MIT | https://github.com/tannerlinsley/react-query/blob/main/LICENSE | tannerlinsley | npm | npm |
| react-router-dom | 6.8.0 | MIT | https://github.com/remix-run/react-router/blob/main/LICENSE | Remix Software <hello@remix.run> | npm | npm |
| react-scripts | 5.0.1 | MIT | https://github.com/facebook/create-react-app/blob/main/LICENSE | N/A | npm | npm |
| recharts | 2.5.0 | MIT | https://github.com/recharts/recharts/blob/main/LICENSE | recharts group | npm | npm |
| typescript | 4.7.4 | Apache-2.0 | https://github.com/Microsoft/TypeScript/blob/main/LICENSE | Microsoft Corp. | npm | npm |
| web-vitals | 2.1.4 | Apache-2.0 | https://github.com/GoogleChrome/web-vitals/blob/main/LICENSE | Philip Walton <philip@philipwalton.com> | npm | npm |

## Notes

- **Source**: Location where the package was downloaded from (PyPI, npm)
- **Distribution Method**: Method used to install the package (pip, npm)
- **License URL**: Link to the package's license information
- Some packages may have missing information if the registry data is incomplete

## License Summary

| License | Count |
|---------|-------|
| MIT | 39 |
| BSD-3-Clause | 5 |
| Apache-2.0 | 5 |
| MIT License | 4 |
| BSD | 4 |
| N/A | 3 |
| BSD License | 2 |
| Apache License, Version 2.0 | 2 |
| Apache Software License | 2 |
| MIT license | 1 |
| Apache 2 | 1 |
| CC0 (copyright waived) | 1 |
| Apache Software License 2.0 | 1 |
| GNU Lesser General Public License v3 (LGPLv3) | 1 |
| Eclipse Public License v2.0 / Eclipse Distribution License v1.0 | 1 |
| new BSD | 1 |
| HPND | 1 |
| GNU AFFERO GPL 3.0 | 1 |
| LICENSE.md | 1 |
| ISC | 1 |
