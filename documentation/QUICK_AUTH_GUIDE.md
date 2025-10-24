# Quick Upstox Authentication - Daily Guide

**Quick Reference**: Run this every day after 3:30 AM IST when your token expires.

---

## Daily Use (30 seconds)

### Step 1: Open Terminal/PowerShell
Press `Win + R`, type `powershell`, press Enter

### Step 2: Navigate to Backend Directory
```powershell
cd C:\Users\ankit\Desktop\PROJECTS\CS301\ml\backend
```

### Step 3: Run Authentication Script
```powershell
python scripts/setup_upstox_oauth.py
```

### Step 4: Enter Your Credentials
When prompted, enter:
- **Client ID**: `your_client_id_here` (get from Upstox Developer Portal)
- **Client Secret**: `your_client_secret_here` (get from Upstox Developer Portal)
- **Redirect URI**: `http://localhost:3000` (or just press Enter for default)

### Step 5: Authorize in Browser
- Browser will open automatically
- Click "Authorize" button
- Window will show "Authorization Successful!"
- Return to terminal

### Step 6: Done!
You'll see:
```
✓ Access token obtained
✓ Tokens saved to JSON cache
✓ API test successful
✓ SETUP COMPLETE!
```

---

## What This Does

1. Opens a local server on port 3000
2. Opens your browser to Upstox authorization page
3. You click "Authorize"
4. Gets your access token
5. Saves it to `backend/_cache/upstox_tokens.json`
6. Tests the token with RELIANCE stock
7. Token valid until tomorrow 3:30 AM IST

---

## Troubleshooting

### ❌ "Address already in use" (Port 3000 busy)
**Solution**: 
```powershell
# Find and kill process using port 3000
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F
```
Then run the script again.

### ❌ Browser doesn't open automatically
**Solution**: 
- Copy the URL shown in terminal
- Paste into your browser manually
- Complete authorization
- Script will detect it automatically

### ❌ "Client ID is required"
**Solution**: You skipped entering Client ID. Just run again and enter it.

### ❌ "Token exchange failed"
**Solution**: 
1. Check your internet connection
2. Verify Client ID and Secret are correct
3. Try running the script again

### ❌ Python not found
**Solution**: Activate virtual environment first:
```powershell
cd C:\Users\ankit\Desktop\PROJECTS\CS301\ml\backend
venv\Scripts\activate
python scripts/setup_upstox_oauth.py
```

---

## Your Credentials

Get your credentials from: https://upstox.com/developer/apps

**Client ID**: 
```
your_client_id_here
```

**Client Secret**: 
```
your_client_secret_here
```

**Redirect URI**: 
```
http://localhost:3000
```

**Note**: Keep these credentials private. The script will also read them from `backend/_cache/upstox_tokens.json` if they're already saved.

---

## Alternative: Quick Command (One Line)

If you're already in the backend directory with venv activated:
```powershell
python scripts/setup_upstox_oauth.py
```

---

## When to Run This

- **Daily**: After 3:30 AM IST when token expires
- **Error**: When you see 401 Unauthorized errors in your app
- **After restart**: If you cleared your cache

---

## Files Modified

- `backend/_cache/upstox_tokens.json` - Token saved here
- Check token expiry in this file to see when you need to re-authenticate

---

## Quick Check: Is My Token Valid?

```powershell
cd C:\Users\ankit\Desktop\PROJECTS\CS301\ml\backend
python -c "import json; f=open('_cache/upstox_tokens.json'); data=json.load(f); print('Token expires:', data['token_expiry']); f.close()"
```

---

## Summary Card

```
┌─────────────────────────────────────┐
│  UPSTOX DAILY AUTH (30 SECONDS)    │
├─────────────────────────────────────┤
│  1. Open PowerShell                 │
│  2. cd C:\...\ml\backend            │
│  3. python scripts/                 │
│     setup_upstox_oauth.py           │
│  4. Enter credentials (see above)   │
│  5. Click "Authorize" in browser    │
│  6. Done!                           │
└─────────────────────────────────────┘
```

Save this guide somewhere easy to find. You can also print it or keep it open in a tab.

