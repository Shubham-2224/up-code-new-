# API Cost Analysis - 2000 Page PDF

## 📊 API Call Breakdown

### For a 2000-Page PDF

Based on the current optimization, here's the API call pattern:

### 1. **Google Vision API - Preprocessing** (Primary)
- **Calls**: **1 per page** = **2,000 calls**
- **When**: Page-level preprocessing (parallel processing)
- **Purpose**: Cache all text annotations for the entire page
- **Optimization**: Uses cached annotations for all cell extractions (no additional calls needed)

### 2. **Google Vision API - Fallback** (Minimal)
- **Calls**: **~0-50 calls** (only if OCR completely fails)
- **When**: Per-cell extraction when:
  - OCR fails AND
  - Cached page annotations unavailable
- **Purpose**: Fallback for individual cell fields
- **Optimization**: Rarely needed due to preprocessing

### 3. **Google Vision API - Page Level** (Minimal)
- **Calls**: **~0-100 calls** (only if cached annotations unavailable)
- **When**: Extracting Booth Center/Address when cached annotations unavailable
- **Purpose**: Extract page-level fields
- **Optimization**: Usually extracted from cached annotations

### 4. **Google Translate API - Transliteration** (Variable)
- **Calls**: **~1-4 per voter record**
- **When**: Transliterating Devanagari text to English
- **Fields**: Name, Relative Name, Booth Center, Booth Address
- **Estimate**: For ~25,000 voters = **~50,000-100,000 calls**

## 💰 Cost Estimation

### Google Vision API Pricing (as of 2024)

**Text Detection (DOCUMENT_TEXT_DETECTION)**:
- **First 1,000 units/month**: FREE
- **1,001 - 5,000,000 units**: **$1.50 per 1,000 units**
- **Per unit**: **$0.0015 per page** (after free tier)

**Calculation for 2,000 pages**:
- First 1,000 pages: **FREE**
- Remaining 1,000 pages: 1,000 × $0.0015 = **$1.50**

### Google Translate API Pricing

**Text Translation**:
- **First 500,000 characters/month**: FREE
- **500,001+ characters**: **$20 per 1 million characters**
- **Per character**: **$0.00002**

**Estimate for 25,000 voters**:
- Average name length: ~20 characters
- Average relative name: ~20 characters
- Average booth center: ~30 characters
- Average booth address: ~50 characters
- **Total per voter**: ~120 characters
- **Total characters**: 25,000 × 120 = **3,000,000 characters**

**Cost**:
- First 500,000 characters: **FREE**
- Remaining 2,500,000 characters: 2.5M × $0.00002 = **$50.00**

## 📈 Total Cost Estimate

### Scenario 1: Optimal (Minimal Fallback)
- **Vision API**: $1.50 (2,000 pages)
- **Translate API**: $50.00 (25,000 voters)
- **Total**: **~$51.50**

### Scenario 2: With Some Fallback
- **Vision API**: $1.50 + $0.08 (50 fallback calls) = **$1.58**
- **Translate API**: $50.00
- **Total**: **~$51.58**

### Scenario 3: Worst Case (More Fallback)
- **Vision API**: $1.50 + $0.15 (100 fallback calls) = **$1.65**
- **Translate API**: $50.00
- **Total**: **~$51.65**

## 🎯 Key Insights

### 1. **Vision API is Very Efficient**
- Only **1 call per page** (not per cell!)
- **2,000 pages = 2,000 calls** (not 200,000+)
- Cost: **~$1.50** for 2,000 pages

### 2. **Translate API is the Main Cost**
- **~$50** for transliteration (largest cost component)
- Can be reduced by:
  - Using local transliteration (AI4Bharat) when available
  - Skipping transliteration for some fields
  - Batch processing

### 3. **Optimization Already in Place**
- ✅ **Parallel processing** (8 pages at once)
- ✅ **Cached annotations** (no duplicate calls)
- ✅ **Smart fallback** (only when needed)

## 💡 Cost Optimization Tips

### 1. **Use Free Tier First**
- Google Vision: First 1,000 pages/month = **FREE**
- Google Translate: First 500,000 characters/month = **FREE**

### 2. **Reduce Transliteration Calls**
- Use local AI4Bharat transliteration when available
- Only transliterate critical fields
- Cache transliteration results

### 3. **Batch Processing**
- Process multiple PDFs in one month to maximize free tier
- Group small PDFs together

### 4. **Monitor API Usage**
- Check logs for actual API call counts
- Look for: `API Calls: Vision=X, Translate=Y, Total=Z`

## 📊 Real-World Example

Based on your logs (1468 pages, 25,412 records):
```
API Calls: Vision=1461, Translate=0, Total=1461
```

**Actual Cost**:
- Vision: 1,461 calls
  - First 1,000: FREE
  - Remaining 461: 461 × $0.0015 = **$0.69**
- Translate: 0 calls (using local transliteration)
- **Total: $0.69** ✅

## 🎯 For 2000-Page PDF Estimate

**Conservative Estimate**:
- **Vision API**: ~$1.50 (2,000 pages)
- **Translate API**: ~$50.00 (if using Google Translate)
- **Total**: **~$51.50**

**If Using Local Transliteration** (like your current setup):
- **Vision API**: ~$1.50
- **Translate API**: $0.00 (local AI4Bharat)
- **Total**: **~$1.50** 🎉

## 📝 Notes

1. **Free Tier**: First 1,000 Vision API calls/month are FREE
2. **Local Transliteration**: Using AI4Bharat reduces Translate API costs to $0
3. **Cached Annotations**: Preprocessing eliminates per-cell API calls
4. **Parallel Processing**: Doesn't affect cost, only speed

## 🔍 How to Check Actual Costs

After processing, check the logs:
```bash
sudo journalctl -u voter-extraction | grep "API Calls"
```

You'll see:
```
API Calls: Vision=2000, Translate=50000, Total=52000
```

Then calculate:
- Vision: (2000 - 1000) × $0.0015 = $1.50
- Translate: (50000 × 20 chars - 500000) × $0.00002 = $X.XX

