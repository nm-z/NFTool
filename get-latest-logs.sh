#!/bin/bash

docker-compose logs --timestamps backend frontend \
  | python3 -c "
import sys, re
from datetime import datetime

lines = sys.stdin.readlines()

traceback_indices = []
for i, line in enumerate(lines):
    if 'Traceback (most recent call last):' in line:
        traceback_indices.append(i)

last_5_starts = traceback_indices[-5:] if len(traceback_indices) >= 5 else traceback_indices

output_lines = set()
for start in last_5_starts:
    for i in range(start, min(start + 52, len(lines))):
        output_lines.add(i)

for i in sorted(output_lines):
    line = lines[i]
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d+)Z', line)
    if match:
        year, month, day, hour, minute, second = match.groups()[:6]
        # Parse as UTC then convert to local
        dt = datetime.fromisoformat(f'{year}-{month}-{day}T{hour}:{minute}:{second}+00:00')
        formatted = dt.astimezone().strftime('%A, %B %-d, %Y %-I:%M %p')
        line = line.replace(match.group(0), formatted)
    print(line, end='')
"