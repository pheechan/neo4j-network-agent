# -*- coding: utf-8 -*-
"""Test bullet formatting fix"""

def fix_bullet_formatting(text: str) -> str:
	"""
	Fix bullet point formatting to ensure each bullet is on a new line.
	Converts inline bullets like '• item1 • item2 • item3' to separate lines.
	"""
	import re
	
	# Split by lines to process each line
	lines = text.split('\n')
	fixed_lines = []
	
	for line in lines:
		# Count bullets in this line
		bullet_count = line.count('•')
		
		if bullet_count > 1:
			# Multiple bullets on same line - split them
			parts = line.split('•')
			if len(parts) > 1:
				# First part before first bullet (usually empty or heading)
				if parts[0].strip():
					fixed_lines.append(parts[0].strip())
				
				# Add each bullet on its own line
				for part in parts[1:]:
					if part.strip():  # Only add if not empty
						fixed_lines.append('• ' + part.strip())
			else:
				fixed_lines.append(line)
		else:
			# Single bullet or no bullet - keep as is
			fixed_lines.append(line)
	
	return '\n'.join(fixed_lines)

# Test with your example
test_text = """อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:

• นายกรัฐมนตรี • รัฐมนตรีว่าการ (ไม่มีข้อมูลกระทรวงในระบบ)"""

print("BEFORE FIX:")
print(test_text)
print("\n" + "="*60 + "\n")

fixed_text = fix_bullet_formatting(test_text)
print("AFTER FIX:")
print(fixed_text)
