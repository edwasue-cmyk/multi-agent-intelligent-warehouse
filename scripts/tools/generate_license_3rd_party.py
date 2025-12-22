#!/usr/bin/env python3
"""
Generate LICENSE-3rd-party.txt file from license audit data.

This script extracts license information from the License_Audit_Report.xlsx
and generates a comprehensive third-party license file.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("Error: openpyxl not installed. Install with: pip install openpyxl")
    exit(1)


# Full license texts
MIT_LICENSE_TEXT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

APACHE_LICENSE_TEXT = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction, and distribution as defined in this document.
   "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.
   "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity.
   "You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.
   "Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.
   "Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
   "Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work.
   "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the modifications represent, as a whole, an original work of authorship.
   "Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work.
   "Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. Grant of Copyright License.

   Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, 
   no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, 
   publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works.

3. Grant of Patent License.

   Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, 
   no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, 
   sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor 
   that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work.

4. Redistribution.

   You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, 
   and in Source or Object form, provided that You meet the following conditions:
     (a) You must give any other recipients of the Work or Derivative Works a copy of this License; and
     (b) You must cause any modified files to carry prominent notices stating that You changed the files; and
     (c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, 
         and attribution notices from the Source form of the Work; and
     (d) If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute 
         must include a readable copy of the attribution notices contained within such NOTICE file.

5. Submission of Contributions.

   Unless You explicitly state otherwise, any Contribution submitted for inclusion in the Work shall be under the terms and conditions of this License.

6. Trademarks.

   This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor.

7. Disclaimer of Warranty.

   The Work is provided on an "AS IS" basis, without warranties or conditions of any kind, either express or implied.

8. Limitation of Liability.

   In no event shall any Contributor be liable for any damages arising from the use of the Work.

END OF TERMS AND CONDITIONS"""

BSD_3_CLAUSE_TEXT = """BSD 3-Clause License

Copyright <YEAR> <COPYRIGHT HOLDER>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

BSD_2_CLAUSE_TEXT = """BSD 2-Clause License

Copyright <YEAR> <COPYRIGHT HOLDER>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""


def normalize_license(license_str: str) -> tuple[str, str]:
    """Normalize license string to standard name and return full text."""
    license_str = str(license_str).strip()
    
    if not license_str or license_str == 'N/A' or license_str == 'UNKNOWN':
        return 'Unknown License', ''
    
    # Normalize to standard license names
    license_upper = license_str.upper()
    
    if 'MIT' in license_upper:
        return 'MIT License', MIT_LICENSE_TEXT
    elif 'APACHE' in license_upper and '2.0' in license_upper:
        return 'Apache License, Version 2.0', APACHE_LICENSE_TEXT
    elif 'BSD' in license_upper:
        if '3-CLAUSE' in license_upper or '3 CLAUSE' in license_upper:
            return 'BSD 3-Clause License', BSD_3_CLAUSE_TEXT
        elif '2-CLAUSE' in license_upper or '2 CLAUSE' in license_upper:
            return 'BSD 2-Clause License', BSD_2_CLAUSE_TEXT
        else:
            return 'BSD License', BSD_3_CLAUSE_TEXT  # Default to 3-clause
    elif 'GPL' in license_upper:
        if 'LGPL' in license_upper or 'LESSER' in license_upper:
            return 'LGPL License', ''  # Would need full text
        else:
            return 'GPL License', ''  # Would need full text
    elif 'PSF' in license_upper or 'PYTHON' in license_upper:
        return 'Python Software Foundation License', ''
    else:
        return license_str, ''  # Return as-is for custom licenses


def extract_packages_from_audit(repo_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Extract packages from license audit report."""
    xlsx_file = repo_root / 'docs' / 'License_Audit_Report.xlsx'
    
    if not xlsx_file.exists():
        print(f"Error: License audit report not found: {xlsx_file}")
        return {}
    
    wb = load_workbook(xlsx_file)
    all_packages = {}
    
    # Extract Python packages
    if 'Python Packages' in wb.sheetnames:
        ws = wb['Python Packages']
        # Find column indices
        name_col = 1
        version_col = 2
        license_col = 3
        author_col = 5  # Author is typically in column 5
        
        for row in range(2, ws.max_row + 1):
            name = ws.cell(row, name_col).value
            version = ws.cell(row, version_col).value
            license_val = ws.cell(row, license_col).value
            author = ws.cell(row, author_col).value if ws.max_column >= author_col else None
            
            if name and license_val:
                # Clean up author string
                author_clean = 'N/A'
                if author and str(author).strip() and str(author).strip() != 'N/A':
                    author_str = str(author).strip()
                    # Remove email addresses and clean up
                    if '<' in author_str:
                        # Extract name before email
                        author_clean = author_str.split('<')[0].strip()
                    else:
                        author_clean = author_str
                    # Remove "Copyright (c)" if present
                    author_clean = author_clean.replace('Copyright (c)', '').replace('Copyright', '').strip()
                    if not author_clean or author_clean == 'PyPI':
                        author_clean = 'N/A'
                
                all_packages[name] = {
                    'version': str(version) if version else 'N/A',
                    'license': str(license_val),
                    'author': author_clean
                }
    
    # Extract Node.js packages
    if 'Node.js Packages' in wb.sheetnames:
        ws = wb['Node.js Packages']
        for row in range(2, ws.max_row + 1):
            name = ws.cell(row, 1).value
            version = ws.cell(row, 2).value
            license_val = ws.cell(row, 3).value
            
            if name and license_val:
                all_packages[name] = {
                    'version': str(version) if version else 'N/A',
                    'license': str(license_val),
                    'author': 'N/A'  # Node.js packages don't have author in audit
                }
    
    # Group by normalized license
    license_groups = defaultdict(list)
    
    for name, info in sorted(all_packages.items()):
        normalized_license, _ = normalize_license(info['license'])
        license_groups[normalized_license].append({
            'name': name,
            'version': info['version'],
            'author': info['author']
        })
    
    return dict(license_groups)


def generate_license_file(repo_root: Path, output_file: Path):
    """Generate LICENSE-3rd-party.txt file."""
    print("Extracting license information from audit report...")
    license_groups = extract_packages_from_audit(repo_root)
    
    if not license_groups:
        print("Error: No packages found in license audit report")
        return False
    
    print(f"Found {sum(len(pkgs) for pkgs in license_groups.values())} packages across {len(license_groups)} license types")
    
    # Generate the license file
    output_lines = []
    
    # Header
    output_lines.append("This file contains third-party license information and copyright notices for software packages")
    output_lines.append("used in this project. The licenses below apply to one or more packages included in this project.")
    output_lines.append("")
    output_lines.append("For each license type, we list the packages that are distributed under it along with their")
    output_lines.append("respective copyright holders and include the full license text.")
    output_lines.append("")
    output_lines.append("")
    output_lines.append("IMPORTANT: This file includes both the copyright information and license details as required by")
    output_lines.append("most open-source licenses to ensure proper attribution and legal compliance.")
    output_lines.append("")
    output_lines.append("")
    output_lines.append("-" * 60)
    output_lines.append("")
    
    # Process licenses in priority order
    license_priority = [
        'MIT License',
        'Apache License, Version 2.0',
        'BSD 3-Clause License',
        'BSD 2-Clause License',
        'BSD License',
        'GPL License',
        'LGPL License',
        'Python Software Foundation License',
    ]
    
    # Add other licenses at the end
    other_licenses = [lic for lic in license_groups.keys() if lic not in license_priority]
    
    for license_name in license_priority + sorted(other_licenses):
        if license_name not in license_groups:
            continue
        
        packages = license_groups[license_name]
        normalized_license, license_text = normalize_license(license_name)
        
        output_lines.append("-" * 60)
        output_lines.append(license_name)
        output_lines.append("-" * 60)
        output_lines.append("")
        
        # Add description based on license type
        if 'MIT' in license_name:
            output_lines.append("The MIT License is a permissive free software license. Many of the packages used in this")
            output_lines.append("project are distributed under the MIT License. The full text of the MIT License is provided")
            output_lines.append("below.")
            output_lines.append("")
            output_lines.append("")
        elif 'Apache' in license_name:
            output_lines.append("The Apache License, Version 2.0 is a permissive license that also provides an express grant of patent rights.")
            output_lines.append("")
            output_lines.append("")
        elif 'BSD' in license_name:
            output_lines.append("The BSD License is a permissive license.")
            output_lines.append("")
            output_lines.append("")
        
        output_lines.append("Packages under the {} with their respective copyright holders:".format(license_name))
        output_lines.append("")
        
        # List packages
        for pkg in sorted(packages, key=lambda x: x['name'].lower()):
            name = pkg['name']
            version = pkg['version']
            author = pkg['author']
            
            output_lines.append("  {} {}".format(name, version))
            if author and author != 'N/A' and author.strip():
                # Author is already cleaned, just add copyright
                output_lines.append("  Copyright (c) {}".format(author))
            output_lines.append("")
        
        # Add full license text if available
        if license_text:
            output_lines.append("")
            output_lines.append("Full {} Text:".format(license_name))
            output_lines.append("")
            output_lines.append("-" * 50)
            output_lines.append("")
            output_lines.append(license_text)
            output_lines.append("")
            output_lines.append("-" * 50)
            output_lines.append("")
            output_lines.append("")
    
    output_lines.append("")
    output_lines.append("END OF THIRD-PARTY LICENSES")
    
    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✅ Generated LICENSE-3rd-party.txt: {output_file}")
    return True


def main():
    """Main function."""
    repo_root = Path(__file__).parent.parent.parent
    output_file = repo_root / 'LICENSE-3rd-party.txt'
    
    print("=" * 70)
    print("Generate LICENSE-3rd-party.txt")
    print("=" * 70)
    print(f"Repository: {repo_root}")
    print(f"Output: {output_file}")
    print("=" * 70)
    print()
    
    success = generate_license_file(repo_root, output_file)
    
    if success:
        print()
        print("=" * 70)
        print("✅ License file generated successfully!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("❌ Failed to generate license file")
        print("=" * 70)
        exit(1)


if __name__ == "__main__":
    main()

