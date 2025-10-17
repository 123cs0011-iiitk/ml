import requests
import json

# Test TATACHEM 5year data
resp = requests.get('http://127.0.0.1:5000/historical?symbol=TATACHEM&period=5year')
data = resp.json()

print('Success:', data.get('success'))
print('Total points:', len(data.get('data', [])))

pts = data.get('data', [])
if pts:
    print('\nFirst 3 points:')
    for p in pts[:3]:
        print(f"  {p['date']}: close={p['close']}, currency={p.get('currency', 'MISSING')}")
    
    print('\nLast 3 points:')
    for p in pts[-3:]:
        print(f"  {p['date']}: close={p['close']}, currency={p.get('currency', 'MISSING')}")

