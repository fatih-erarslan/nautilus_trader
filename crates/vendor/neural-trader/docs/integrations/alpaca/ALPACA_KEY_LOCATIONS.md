# 🔍 Where to Find Your Alpaca Secret Key

## Dashboard Locations to Check:

### 1. **New Dashboard (app.alpaca.markets)**
- Go to: https://app.alpaca.markets/paper/dashboard/overview
- Look for **"API Keys"** in the left sidebar menu
- Or click your profile icon → **"API Keys"**

### 2. **Direct API Keys Page**
Try these direct links:
- https://app.alpaca.markets/paper/dashboard/api-keys
- https://app.alpaca.markets/brokerage/dashboard/api_management
- https://alpaca.markets/deprecated/paper/account/api-keys

### 3. **If You See Only the API Key (PKVZM47F4PZC9B4QB3KF)**
The secret key may be:
- Hidden behind a **"Show"** or **"Reveal"** button
- Already expired (you'll see "Regenerate" button)
- Not displayed (need to regenerate)

## 🔄 To Generate New Keys:

### Option 1: Regenerate Existing Key
1. Find your key `PKVZM47F4PZC9B4QB3KF` in the dashboard
2. Click **"Regenerate"** or **"Reset"** button
3. **IMMEDIATELY COPY** both keys shown in the popup

### Option 2: Create Fresh Keys
1. Click **"Generate New API Key"** or **"+ New"**
2. Name it: "Neural Trader"
3. Select: **Paper Trading** (not Live)
4. Copy both:
   - **Key ID**: (like PKxxxxx...)
   - **Secret Key**: (long random string)

## 🎯 What You're Looking For:

The secret key will look like one of these formats:
- Format 1: `RkG7nB3mPQX9YuL5Kw2Jv8NcT1MhZsA4Df6Eq0Ip`
- Format 2: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0`
- Format 3: Base64 encoded string with +/= characters

## 📱 Mobile App Alternative:
If you have the Alpaca mobile app:
1. Open the app
2. Go to Settings → API Management
3. View or regenerate paper trading keys

## 🆘 If Nothing Works:

**Create brand new keys right now:**

1. Go here: https://app.alpaca.markets/paper/dashboard/overview
2. Find ANY button that says:
   - "API"
   - "Keys"
   - "Developer"
   - "Generate"
3. Click it and generate new paper trading keys
4. You'll get:
   - New API Key ID (replace PKVZM47F4PZC9B4QB3KF)
   - New Secret Key (the missing piece we need)

## 💡 Quick Test:
Once you have BOTH keys, test with:
```bash
curl -X GET \
  -H "APCA-API-KEY-ID: your-key-id" \
  -H "APCA-API-SECRET-KEY: your-secret" \
  https://paper-api.alpaca.markets/v2/clock
```

Should return current market time if successful!