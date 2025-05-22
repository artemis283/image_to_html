#!/usr/bin/env python
import os
import sys

if len(sys.argv) < 2:
    print("Usage: check-file.py <html_file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# Read file
with open(file_path, 'r') as f:
    content = f.read()

# Check actual line count
lines = content.splitlines()
print(f"File: {os.path.basename(file_path)}")
print(f"Total characters: {len(content)}")
print(f"Total lines: {len(lines)}")

# Check line lengths
long_lines = 0
very_long_lines = 0
for line in lines:
    if len(line) > 200:
        long_lines += 1
    if len(line) > 1000:
        very_long_lines += 1

print(f"Lines over 200 chars: {long_lines}")
print(f"Lines over 1000 chars: {very_long_lines}")

# Show longest line
if lines:
    longest = max(lines, key=len)
    print(f"\nLongest line has {len(longest)} characters")
    print(f"Preview: {longest[:100]}...")

# Count actual newlines
newline_count = content.count('\n')
print(f"\nActual newline characters: {newline_count}")

# Check if it's all on one line
if len(lines) < 10 and len(content) > 1000:
    print("\n⚠️  WARNING: File might be minified (all on few lines)")
    print("First 500 chars:")
    print(content[:500])
    print("\n... [truncated] ...")
    print("\nLast 500 chars:")
    print(content[-500:])