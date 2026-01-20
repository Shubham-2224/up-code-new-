# Scalability Guide - Single User vs Multi-User

## 🔍 Current Setup Analysis

### Current Configuration
- **Server**: Flask development server (`app.run()`)
- **Threading**: Single-threaded (default Flask)
- **Storage**: In-memory dictionaries (not persistent)
- **State Management**: No user sessions or isolation
- **Resource Usage**: CPU-intensive (OCR, image processing)

## ⚠️ Current Limitations

### 1. **Single-Threaded Server**
- Flask's built-in server handles **one request at a time**
- Multiple users will **queue up** and wait
- Not suitable for production with multiple users

### 2. **In-Memory Storage**
- All data stored in Python dictionaries
- **Lost on restart**
- **Shared between all users** (no isolation)
- Risk of data conflicts

### 3. **Resource Intensive**
- Each extraction uses:
  - CPU: All cores for parallel processing
  - Memory: 500MB+ per large PDF
  - Disk: Temporary files
- Multiple concurrent extractions could **crash the server**

## 📊 User Capacity Analysis

### Current Setup (Single-Threaded Flask)

| Scenario | Users | Status | Notes |
|----------|-------|--------|-------|
| **Single User** | 1 | ✅ **OK** | Works perfectly |
| **2-3 Users** | 2-3 | ⚠️ **Slow** | Requests queue, long wait times |
| **5+ Users** | 5+ | ❌ **Not Recommended** | Server may crash, data conflicts |

### Recommended: **Single User or 2-3 Users Maximum**

## 🎯 Recommendations

### Option 1: Single User (Recommended for Now)
**Best for:**
- Personal/internal use
- Low traffic
- Testing/development
- Cost-effective

**Configuration:**
- Current setup is fine
- No changes needed
- Add rate limiting to prevent abuse

### Option 2: Multi-User (Requires Upgrades)
**Best for:**
- Public deployment
- Multiple concurrent users
- Production environment

**Required Changes:**
1. Production WSGI server (Gunicorn)
2. Database for state management
3. User authentication/sessions
4. Rate limiting
5. Resource management
6. Queue system for heavy operations

## 🚀 Quick Fix: Support 2-3 Users

### Minimal Changes Needed

#### 1. **Add Gunicorn** (Production Server)

Install:
```bash
pip install gunicorn
```

Update service file:
```ini
[Service]
ExecStart=/path/to/venv/bin/gunicorn \
    --workers 2 \
    --threads 2 \
    --bind 0.0.0.0:5000 \
    --timeout 1200 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    app:app
```

**Benefits:**
- Handles 2-4 concurrent requests
- Better performance
- Production-ready

**Limitations:**
- Still uses in-memory storage (data lost on restart)
- No user isolation
- Resource conflicts possible

#### 2. **Add Rate Limiting**

Install:
```bash
pip install flask-limiter
```

Add to `app.py`:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route('/api/upload-pdf', methods=['POST'])
@limiter.limit("5 per minute")  # Max 5 uploads per minute per IP
def upload_pdf():
    # ... existing code ...
```

## 📈 Full Multi-User Setup

### Required Components

#### 1. **Production Server (Gunicorn)**
```bash
pip install gunicorn
```

Service configuration:
```ini
[Service]
ExecStart=/path/to/venv/bin/gunicorn \
    --workers 4 \
    --threads 2 \
    --bind 0.0.0.0:5000 \
    --timeout 1200 \
    --worker-class gthread \
    app:app
```

#### 2. **Database for State Management**

Replace in-memory dictionaries with database:

**Option A: SQLite** (Simple)
```python
import sqlite3

# Replace uploaded_files dict with database
def init_db():
    conn = sqlite3.connect('extraction.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            file_id TEXT PRIMARY KEY,
            user_id TEXT,
            filepath TEXT,
            created_at TIMESTAMP
        )
    ''')
```

**Option B: PostgreSQL** (Production)
- Better for multiple users
- Handles concurrent access
- Persistent storage

#### 3. **User Authentication**

Add user sessions:
```python
from flask_session import Session

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
```

#### 4. **Resource Management**

Limit concurrent extractions:
```python
MAX_CONCURRENT_EXTRACTIONS = 2  # Only 2 extractions at a time

extraction_semaphore = threading.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

@app.route('/api/extract-grid', methods=['POST'])
def extract_grid():
    if not extraction_semaphore.acquire(blocking=False):
        return jsonify({'error': 'Server busy, please try again later'}), 503
    try:
        # ... extraction code ...
    finally:
        extraction_semaphore.release()
```

#### 5. **Queue System** (For Heavy Operations)

Use Celery or Redis Queue:
```python
from celery import Celery

celery = Celery('extractor', broker='redis://localhost:6379')

@celery.task
def extract_task(pdf_bytes, config):
    # Heavy extraction work
    return result
```

## 💰 Cost Considerations

### Single User Setup
- **EC2 Instance**: t3.medium (2 vCPU, 4GB RAM) - ~$30/month
- **Storage**: 20GB EBS - ~$2/month
- **Total**: **~$32/month**

### Multi-User Setup (10 users)
- **EC2 Instance**: t3.large (4 vCPU, 8GB RAM) - ~$60/month
- **Database**: RDS PostgreSQL (db.t3.micro) - ~$15/month
- **Storage**: 50GB EBS - ~$5/month
- **Total**: **~$80/month**

## 🎯 My Recommendation

### For Your Current Use Case:

**Start with Single User Setup** ✅

**Reasons:**
1. **Current code is optimized for single user**
2. **Resource-intensive** (uses all CPU cores)
3. **In-memory storage** (simple, fast)
4. **Cost-effective** (smaller instance)
5. **Easier to manage**

**When to Upgrade to Multi-User:**
- When you have **consistent demand** from multiple users
- When you need **data persistence** across restarts
- When you have **budget for larger instance + database**

### Quick Multi-User Support (2-3 Users)

**Minimal changes:**
1. Install Gunicorn
2. Update service file (2 workers)
3. Add rate limiting
4. **Cost**: Same (~$32/month)
5. **Capacity**: 2-3 concurrent users

## 📝 Implementation Steps

### For Single User (Current - No Changes)
✅ Already configured correctly
✅ Just add rate limiting for security

### For 2-3 Users (Quick Upgrade)

1. **Install Gunicorn**:
   ```bash
   cd backend/python-service
   source venv/bin/activate
   pip install gunicorn
   ```

2. **Update service file** (I can create this for you)

3. **Add rate limiting** (I can add this)

4. **Restart service**:
   ```bash
   sudo systemctl restart voter-extraction
   ```

### For Full Multi-User (Production)

Requires significant refactoring:
- Database integration
- User authentication
- Queue system
- Resource management

**Estimated effort**: 2-3 days of development

## 🔒 Security Considerations

### Current Setup (Single User)
- ✅ CORS configured
- ✅ File size limits
- ✅ Path traversal protection
- ⚠️ No rate limiting
- ⚠️ No authentication

### Multi-User Setup (Required)
- ✅ All single-user protections
- ✅ Rate limiting
- ✅ User authentication
- ✅ Session management
- ✅ Resource quotas per user

## 📊 Summary

| Setup | Users | Cost/Month | Complexity | Recommended For |
|-------|-------|------------|------------|-----------------|
| **Current** | 1 | $32 | Low | Personal use, testing |
| **Quick Upgrade** | 2-3 | $32 | Low | Small team |
| **Full Multi-User** | 10+ | $80+ | High | Production, public |

## 🎯 My Answer

**For now: Keep it Single User** ✅

**Reasons:**
- Your current setup is perfect for single user
- Resource-intensive operations work best with dedicated resources
- Simpler to manage and debug
- Cost-effective

**When you need multi-user:**
- Start with quick upgrade (Gunicorn + 2 workers)
- Then add database if you need persistence
- Then add authentication if you need user isolation

Would you like me to:
1. **Add Gunicorn** for 2-3 user support?
2. **Add rate limiting** for security?
3. **Create full multi-user setup** guide?


