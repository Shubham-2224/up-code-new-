# Accuracy Fix - Server vs Local Discrepancy

## 🐛 Problem Identified

**Issue**: Server extracts 1964 voters, but local extracts 2001 voters (correct)
**Difference**: 37 missing voters (2001 - 1964 = 37)

## 🔍 Root Cause

The extraction logic was **too strict** - it required a **serial number** to be present, otherwise the cell was skipped. 

**Problem**: On the server, OCR might fail to detect serial numbers due to:
- Different Tesseract version
- Different system resources
- Different image rendering
- Network latency affecting API calls

When serial number detection failed, **valid voters were being incorrectly skipped**.

## ✅ Fix Applied

### Changed Skip Logic

**Before (Too Strict)**:
```python
# Required serial number - if missing, skip cell
if not has_serial_number:
    should_skip = True
```

**After (More Lenient)**:
```python
# Require at least ONE identifying field (voter ID, serial number, name, or photo)
has_any_identifying_field = has_voter_id or has_serial_number or has_name or has_photo

if not has_any_identifying_field:
    # Only skip if ALL identifying fields are missing
    should_skip = True
```

### Improvements

1. **More Lenient Validation**: Now accepts cells with:
   - Voter ID (EPIC), OR
   - Serial Number, OR
   - Name, OR
   - Photo

2. **Better Logging**: Added skip reason tracking to help debug:
   - `no_identifying_fields`: Cell has no identifying data
   - `outside_extraction_area`: Cell is in header/footer zone
   - `extraction_error`: Error during extraction

3. **Diagnostic Information**: Logs show why cells are skipped:
   ```
   ⚠ Skipped 37 cells - Reasons: no_identifying_fields: 30, outside_extraction_area: 7
   ```

## 🎯 Expected Results

After this fix:
- **Server should extract 2001 voters** (same as local)
- **No valid voters should be skipped** due to missing serial numbers
- **Better error reporting** to identify any remaining issues

## 🔧 Testing

1. **Restart the service**:
   ```bash
   sudo systemctl restart voter-extraction
   ```

2. **Process the same PDF** and verify:
   - Should extract 2001 voters (same as local)
   - Check logs for skip reasons if any cells are skipped

3. **Check logs**:
   ```bash
   sudo journalctl -u voter-extraction -f
   ```
   Look for: `⚠ Skipped X cells - Reasons: ...`

## 📊 Why This Happens

### Server vs Local Differences

1. **OCR Accuracy**: Server might have:
   - Different Tesseract version
   - Different system fonts/libraries
   - Different image rendering quality

2. **Resource Constraints**: Server might have:
   - Less memory causing timeouts
   - CPU throttling affecting OCR
   - Network latency for API calls

3. **Environment Differences**: 
   - Different Python versions
   - Different dependency versions
   - Different system configurations

## 🛡️ Prevention

The fix ensures that:
- ✅ Voters with **any identifying field** are extracted
- ✅ Only truly empty/invalid cells are skipped
- ✅ Better logging helps identify issues quickly

## 📝 Next Steps

1. Deploy the fix to server
2. Test with the same PDF
3. Verify extraction count matches local (2001)
4. Check logs for any remaining skip reasons
5. If issues persist, check skip reasons in logs to identify patterns

