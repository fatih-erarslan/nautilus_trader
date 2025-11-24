#!/usr/bin/env python3
"""
Fix datetime.utcnow() deprecation warnings in messaging files
"""

import os
import re

def fix_datetime_in_file(filename):
    """Fix datetime.utcnow() usage in a file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Replace datetime.utcnow() with datetime.now(datetime.timezone.utc)
        # But first check if datetime is imported and add timezone if needed
        
        # Check if timezone is imported
        if 'from datetime import' in content and 'timezone' not in content:
            # Add timezone to existing import
            content = re.sub(
                r'from datetime import ([^,\n]+)',
                r'from datetime import \1, timezone',
                content
            )
        elif 'import datetime' in content and 'timezone' not in content:
            # Add timezone import
            content = re.sub(
                r'import datetime',
                r'import datetime\nfrom datetime import timezone',
                content
            )
        
        # Replace utcnow() calls
        content = re.sub(
            r'datetime\.utcnow\(\)',
            r'datetime.now(timezone.utc)',
            content
        )
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed {filename}")
        
    except Exception as e:
        print(f"✗ Error fixing {filename}: {e}")

# Fix messaging files
files_to_fix = [
    'pads_messaging_integration.py',
    'quantum_amos_messaging_adapter.py', 
    'quasar_messaging_adapter.py',
    'test_integration.py'
]

for filename in files_to_fix:
    if os.path.exists(filename):
        fix_datetime_in_file(filename)
    else:
        print(f"File not found: {filename}")

print("✓ Datetime fixes complete")