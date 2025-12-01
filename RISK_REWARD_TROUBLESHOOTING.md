# Risk-Reward Feature Troubleshooting Guide

## How to Access the Risk-Reward Feature

1. **Open the Analysis Tool (Chat with Caria)**
   - Click "Ask Caria" widget on the Dashboard
   - Or use the "Start Analysis" button

2. **Look for the "Show Risk-Reward" Button**
   - Located in the top-right of the Analysis Tool header
   - Next to the close (X) button
   - Should be visible by default (blue/active state)

3. **The Panel Should Appear**
   - **Desktop**: Right side (50% width) next to the chat
   - **Mobile**: Below the chat (full width, scroll down)

## If You Can't See It

### Check 1: Is the Button Visible?
- Look for a button labeled "Show Risk-Reward" or "Hide Risk-Reward" in the Analysis Tool header
- If you see "Hide Risk-Reward", the panel is already shown (click to toggle)

### Check 2: On Mobile - Scroll Down
- On mobile devices, the Risk-Reward panel appears **below** the chat
- Scroll down in the Analysis Tool to see it

### Check 3: Browser Cache
- Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
- Or clear browser cache and reload

### Check 4: Frontend Rebuild Required
- The feature requires a frontend rebuild
- If using Vercel, it should auto-deploy from the git push
- Check deployment status in Vercel dashboard

### Check 5: Check Browser Console
- Open Developer Tools (F12)
- Look for any errors in the Console tab
- Check Network tab for failed API calls to `/api/risk-reward/analyze`

## Testing the Feature

1. **Open Analysis Tool**
2. **Enter a ticker** in the Risk-Reward panel (e.g., "AAPL")
3. **Wait for analysis** (should load automatically)
4. **Adjust probabilities** using the sliders
5. **View results**:
   - Scenario chart (Bear/Base/Bull bars)
   - Metrics (Upside, Downside, RRR, EV)
   - Educational explanations

## Expected Behavior

- **Panel visible by default** when Analysis Tool opens
- **Button shows "Hide Risk-Reward"** when panel is visible
- **Button shows "Show Risk-Reward"** when panel is hidden
- **Ticker auto-populates** when mentioned in chat (e.g., "What about AAPL?")
- **Real-time updates** when probabilities change

## API Endpoint

The backend endpoint should be available at:
- `POST /api/risk-reward/analyze`

Test with:
```bash
curl -X POST https://your-backend-url/api/risk-reward/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"ticker": "AAPL", "horizon_months": 24}'
```

## Common Issues

### Issue: Panel doesn't appear
**Solution**: Click "Show Risk-Reward" button in header

### Issue: "404 Not Found" for API
**Solution**: 
- Verify backend is deployed
- Check that `backend/api/routes/risk_reward.py` exists
- Verify route is registered in `backend/api/domains/analysis/routes.py`

### Issue: Chart shows "width/height -1" error
**Solution**: Fixed in latest commit - rebuild frontend

### Issue: Panel is empty
**Solution**: 
- Enter a ticker symbol (e.g., "AAPL")
- Wait for API response
- Check browser console for errors

## Files Changed

- `frontend/caria-app/components/RiskRewardPanel.tsx` - Main panel component
- `frontend/caria-app/components/AnalysisTool.tsx` - Integration
- `backend/api/services/risk_reward_service.py` - Backend service
- `backend/api/routes/risk_reward.py` - API endpoint
- `backend/api/domains/analysis/routes.py` - Route registration

## Next Steps if Still Not Working

1. **Verify Deployment**: Check that latest code is deployed
2. **Check Logs**: Look at backend logs for API errors
3. **Test API Directly**: Use Postman/curl to test the endpoint
4. **Check Network Tab**: See if requests are being made
5. **Verify Imports**: Ensure RiskRewardPanel is imported correctly

