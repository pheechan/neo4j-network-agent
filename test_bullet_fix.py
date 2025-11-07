import re

def fix_bullet_formatting(text: str) -> str:
	"""
	Fix bullet point formatting to ensure each bullet is on a new line.
	Converts inline bullets like '• item1 • item2 • item3' to separate lines.
	"""
	# Split by lines to process each line
	lines = text.split('\n')
	fixed_lines = []
	
	for line in lines:
		# Count bullets in this line
		bullet_count = line.count('•')
		
		if bullet_count > 1:
			# Multiple bullets on same line - split them
			# Replace ' •' with '\n•' but not the first bullet
			parts = line.split('•')
			if len(parts) > 1:
				# First part before first bullet
				result = parts[0]
				# Add each bullet on new line
				for i, part in enumerate(parts[1:], 1):
					if part.strip():  # Only add if not empty
						result += '•' + part.rstrip()
						if i < len(parts) - 1:  # Not the last one
							result += '\n'
				fixed_lines.append(result)
			else:
				fixed_lines.append(line)
		else:
			# Single bullet or no bullet - keep as is
			fixed_lines.append(line)
	
	return '\n'.join(fixed_lines)

# Test with exact format from user's output
test_text = """บุคคลที่เกี่ยวข้อง: • พี่โด่ง - รัฐมนตรีประจำสำนักงานปลัด • พี่บอย (CGC) - รัฐมนตรี • อภัชัย (IOD, วตท) - รัฐมนตรี"""

print("BEFORE:")
print(test_text)
print("\n" + "="*60 + "\n")

fixed = fix_bullet_formatting(test_text)
print("AFTER:")
print(fixed)
