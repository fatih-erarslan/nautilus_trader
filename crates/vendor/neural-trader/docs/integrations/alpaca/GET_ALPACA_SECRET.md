# 🔑 How to Get Your Alpaca Secret Key

## Step 1: Log into Alpaca
Go to: https://app.alpaca.markets/paper/dashboard/overview

## Step 2: Navigate to API Keys
Look for one of these options in your dashboard:
- **"API Keys"** section
- **"Your API Keys"**
- **"Paper Trading API"**
- A key icon 🔑

## Step 3: View or Generate Keys

### Option A: If you see your keys listed:
- You should see:
  - **Key ID**: PKVZM47F4PZC9B4QB3KF (this is what we have)
  - **Secret Key**: [HIDDEN] with a "View" or "Show" button
- Click **"View"** or **"Show"** to reveal the secret key

### Option B: If you don't see a secret key:
1. Click **"Regenerate"** or **"Generate New Key"**
2. A popup will show BOTH keys:
   - API Key ID: PKVZM47F4PZC9B4QB3KF
   - Secret Key: [A long string of characters]
3. **IMPORTANT**: Copy the secret key immediately! It won't be shown again.

## Step 4: What the Secret Key Looks Like
The secret key will be a long string like:
- Example format: `AbCdEfGhIjKlMnOpQrStUvWxYz1234567890+/=`
- Usually 40-50 characters long
- Contains letters, numbers, and sometimes special characters

## Step 5: Alternative - Create New API Keys
If you can't find the secret for PKVZM47F4PZC9B4QB3KF:
1. Click **"Create New API Key"** or **"+ New Key"**
2. Give it a name like "Neural Trader"
3. Copy BOTH the new API Key ID and Secret Key
4. Update both in the .env file

## 🚨 Important Notes:
- The secret key is only shown ONCE when created
- If you've lost it, you need to regenerate the key pair
- Paper trading keys are different from live trading keys
- Make sure you're in the "Paper Trading" section, not "Live Trading"

## Quick Check URL:
Direct link to API management (if logged in):
https://app.alpaca.markets/brokerage/api-keys

Or try the old dashboard:
https://alpaca.markets/deprecated/docs/api-documentation/api-v2/oauth/

---

Once you have the secret key, update line 103 in .env:
```
ALPACA_SECRET_KEY=your-actual-secret-key-here
```