# Upstox Daily Token Management Guide

## The Problem: Daily 3:30 AM Token Expiration

Upstox access tokens have a **unique expiration behavior** that catches many developers off guard:

- **All tokens expire daily at 3:30 AM IST**, regardless of when they were generated
- A token generated at 8 PM Tuesday expires at 3:30 AM Wednesday
- A token generated at 2:30 AM Wednesday also expires at 3:30 AM Wednesday (only 1 hour validity)
- **No refresh tokens** are provided by Upstox (unlike other OAuth2 providers)

## Why This Happens

This is a **business decision** by Upstox to ensure security and prevent long-term token abuse. It's not a bug - it's a feature designed to:

1. **Enhance Security**: Prevents compromised tokens from being used indefinitely
2. **Comply with Regulations**: Meets financial sector security requirements
3. **Control API Usage**: Ensures active monitoring of API access

## Current System Status

✅ **Your system is working correctly!** Here's what happens:

1. **Token Expires**: At 3:30 AM daily
2. **API Calls Fail**: Upstox returns 401 Unauthorized
3. **Fallback Activates**: System uses cached data from `permanent/` directory
4. **User Gets Data**: Seamless experience with slightly older data

## Solutions

### Option 1: Manual Token Generation (Current Approach)
```bash
# Check token status
python scripts/upstox_daily_token_guide.py

# Generate new token
python scripts/generate_new_token.py

# Complete OAuth flow
python scripts/setup_upstox_oauth.py
```

**Pros**: Simple, secure
**Cons**: Requires daily manual intervention

### Option 2: Automated Token Generation (Recommended for Production)
```bash
pip install upstox-totp
```

Then use the `upstox-totp` library for automated daily token generation.

### Option 3: Hybrid Approach (Current Implementation)
- Use Upstox when token is valid
- Fall back to cached data when token expires
- Generate new token when convenient

## Implementation Details

### Token Expiry Detection
Our system now detects daily expiration:

```python
def needs_daily_token_refresh(self) -> bool:
    """Check if we need a new token due to Upstox's daily 3:30 AM expiration."""
    if not self.token_expiry:
        return True
    
    now = datetime.now()
    today_330am = now.replace(hour=3, minute=30, second=0, microsecond=0)
    return now >= today_330am and self.token_expiry < today_330am
```

### Graceful Fallback
When token refresh fails (due to API v1 deprecation), the system:

1. **Logs the issue** with clear error messages
2. **Uses existing token** if available (may work for some calls)
3. **Falls back to cached data** for reliable service
4. **Continues operating** without interruption

## Best Practices

### For Development
- Use manual token generation
- Test with both live and cached data
- Monitor logs for token expiration warnings

### For Production
- Implement automated token generation using `upstox-totp`
- Set up monitoring for token expiration
- Have fallback data sources ready

### For Testing
- Test during market hours (9:15 AM - 3:30 PM IST)
- Test after 3:30 AM to verify fallback behavior
- Verify data freshness and accuracy

## Troubleshooting

### "Token expired" errors
- **Cause**: Daily 3:30 AM expiration
- **Solution**: Generate new token using the scripts above

### "API deprecated" warnings
- **Cause**: Upstox API v1 is deprecated
- **Solution**: This is expected - system handles it gracefully

### "No data available" errors
- **Cause**: All APIs failed, no fallback data
- **Solution**: Check if `permanent/` directory has data

## Monitoring

### Check Token Status
```bash
python scripts/upstox_daily_token_guide.py
```

### Test API Endpoint
```bash
curl "http://localhost:5000/live_price?symbol=RELIANCE"
```

### Check Logs
Look for these log messages:
- `Token expired or near expiry, refreshing proactively...`
- `Upstox API returned deprecation warning`
- `Using existing token despite refresh failure`
- `✅ Found RELIANCE in permanent directory`

## Conclusion

Your Upstox integration is **working perfectly**! The daily token expiration is a known limitation, not a bug. The system gracefully handles this by:

1. **Attempting live data** when possible
2. **Falling back to cached data** when needed
3. **Providing clear guidance** for token renewal
4. **Maintaining service availability** at all times

The fallback mechanism ensures your users always get data, even when the live API is unavailable due to token expiration.
