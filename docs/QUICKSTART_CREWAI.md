# Quick Start: CrewAI Agentic AI Setup

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Get Gemini API Key (Recommended - Free)

1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

### Add to .env

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Test the System

### 1. Start Server

```bash
python backend/manage.py serve
```

### 2. Test High Confidence (Fast Path - No Agents)

```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "What is your name?",
    "text2": "May I know your name?",
    "use_agent": true
  }'
```

**Expected Output**:
```json
{
  "similarity": 0.91,
  "is_paraphrase": true,
  "inference_time_ms": 42.3,
  "agent_validation": false,
  "confidence_level": "HIGH"
}
```

⚡ **Fast**: Agents didn't activate (HIGH confidence)

---

### 3. Test Paraphrase Rescue (Agents Activate)

```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "OK",
    "text2": "That sounds good to me, I completely agree",
    "use_agent": true
  }'
```

**Expected Output**:
```json
{
  "similarity": 0.90,
  "is_paraphrase": true,
  "inference_time_ms": 3250.5,
  "agent_validation": true,
  "confidence_level": "UNCERTAIN",
  "edge_cases": ["length_mismatch", "short_text", "low_confidence_paraphrase"],
  "agent_reasoning": "DECISION: YES\nCONFIDENCE: HIGH\nREASONING: Both texts express agreement...",
  "agent_confidence": "HIGH",
  "paraphrase_rescued": true,
  "original_similarity": 0.45,
  "adjusted_similarity": 0.90
}
```

🎯 **Success**: Agents caught a paraphrase the model missed!
- Model: 0.45 → "Not Paraphrase" ❌
- Agents: HIGH confidence → "Paraphrase" ✅
- `paraphrase_rescued: true`

---

### 4. Test Edge Case (Borderline)

```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Hi",
    "text2": "Hello!",
    "use_agent": true
  }'
```

**Expected Output**:
```json
{
  "similarity": 0.88,
  "is_paraphrase": true,
  "agent_validation": true,
  "edge_cases": ["short_text", "borderline_case"],
  "agent_confidence": "HIGH",
  "paraphrase_rescued": false
}
```

✅ **Validated**: Agents confirmed model's decision with edge cases present

---

## Check Performance Stats

```bash
curl http://localhost:8000/inference/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "model_ready": true,
  "performance": {
    "total_requests": 100,
    "cache_hit_rate": "15.00%",
    "agent_validations": 5,
    "agentic_ai": {
      "enabled": true,
      "agent_activations": 7,
      "activation_rate": "7.00%",
      "paraphrase_rescues": 2,
      "rescue_rate": "28.57%",
      "edge_cases_detected": {
        "length_mismatch": 2,
        "short_text": 3,
        "borderline_case": 4,
        "low_confidence_paraphrase": 2
      }
    }
  }
}
```

**Key Metrics**:
- **Activation Rate**: ~4-7% (agents only when needed)
- **Rescue Rate**: ~20-30% (of activations)
- **Enabled**: true (CrewAI working)

---

## Troubleshooting

### Issue: "CrewAI not enabled" message

**Check**:
```bash
# .env should have:
GEMINI_API_KEY=your_key_here
# OR
OPENAI_API_KEY=your_key_here
```

**Fix**: Add API key and restart server

---

### Issue: Agents never activate

**Check**:
- Using `use_agent: true` in request?
- Testing with borderline cases (similarity 0.55-0.75)?
- API key valid?

**Test with guaranteed activation**:
```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "OK",
    "text2": "I agree completely",
    "use_agent": true
  }'
```

This will definitely activate agents (low similarity + edge cases).

---

### Issue: Slow responses

**Expected**:
- 93-96% of queries: <50ms (fast path, no agents)
- 4-7% of queries: 2-4s (agents activated)

**Check**:
```json
{
  "agent_validation": true  // Slow because agents ran
}
```

This is normal for edge cases! Agents only activate when needed.

**To disable agents** (for speed-critical apps):
```json
{
  "use_agent": false
}
```

---

## Next Steps

1. ✅ Test all 3 scenarios above
2. ✅ Check `/inference/health` stats
3. ✅ Verify `paraphrase_rescued: true` appears
4. 📖 Read [full documentation](AGENTIC_AI.md)
5. 🚀 Deploy to production

---

## Key Commands

```bash
# Start server
python backend/manage.py serve

# Test paraphrase rescue
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{"text1": "OK", "text2": "I agree", "use_agent": true}'

# Check stats
curl http://localhost:8000/inference/health

# View docs
http://localhost:8000/docs
```

---

## Success Indicators

✅ **System Working**:
- `"enabled": true` in `/health` response
- `agent_validation: true` for borderline cases
- `paraphrase_rescued: true` appears occasionally
- `activation_rate: "4-7%"`
- `rescue_rate: "20-30%"`

❌ **System Not Working**:
- `"enabled": false` → Check API key
- `agent_validation: false` always → Check test cases
- Errors in logs → Check dependencies

---

## Support

- 📖 Full docs: [AGENTIC_AI.md](AGENTIC_AI.md)
- 🔧 Implementation details: [CREWAI_IMPLEMENTATION.md](CREWAI_IMPLEMENTATION.md)
- 🚀 General setup: [../README.md](../README.md)
