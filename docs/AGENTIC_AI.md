# CrewAI-Powered Agentic Pipeline for Edge Case Handling

## Overview

The paraphrase detection system includes an intelligent **CrewAI Multi-Agent Pipeline** that automatically detects and handles edge cases using specialized AI agents. This ensures robust predictions even for challenging text comparisons, **especially catching paraphrases the model might miss**.

## Why CrewAI?

**CrewAI** is an advanced multi-agent orchestration framework that enables:
- **Specialized Agents**: Each agent has a specific expertise (paraphrase detection, edge cases, semantic validation)
- **Collaborative Intelligence**: Agents work together sequentially to make better decisions
- **Role-Based Analysis**: Different perspectives ensure comprehensive evaluation
- **Smart Triggering**: Only activates when the model needs help (low confidence, edge cases, borderline decisions)

## Multi-Agent Architecture

```
User Query
    ↓
Siamese Model Prediction
    ↓
Confidence Assessment + Edge Case Detection
    ↓
Smart Trigger Decision
    ↓
[High Confidence + No Edge Cases] → Direct Response (Fast)
    ↓
[Low Confidence OR Edge Cases OR Borderline] → CrewAI Agents Activated
    ↓
┌───rewAI Agents

### Agent 1: Edge Case Specialist 🔍
**Role**: Edge Case Detection Specialist

**Expertise**:
- Identifies length mismatches, short texts, numeric-heavy content
- Detects special characters and formatting anomalies
- Explains why each edge case might confuse the ML model

**Example Analysis**:
```
"Text A is 2 words while Text B is 15 words - significant length mismatch.
This can cause the model to underestimate similarity. Additionally, both
texts are relatively short, reducing available context for embedding."
```

### Agent 2: Semantic Validator 🧠
**Role**: Semantic Equivalence Validator

**Expertise**:
### Model Confidence Classification

The system classifies predictions into 4 confidence levels:

| Level | Similarity Range | Model Behavior | Agent Trigger |
|-------|------------------|----------------|---------------|
| **HIGH** | > 0.85 | Trust model completely | ❌ No (fast path) |
| **MEDIUM** | 0.70 - 0.85 | Likely correct, monitor | ✅ If edge cases |
| **LOW** | 0.55 - 0.70 | Uncertain, validate | ✅ Yes (borderline) |
| **UNCERTAIN** | < 0.55 | Low confidence | ✅ Yes (always) |

### Smart Triggering Logic7

**Agents activate when**:
1. **Model Uncertain** (similarity < 0.55) - Always activate
2. **Borderline Cases** (0.55-0.75) - **Key for catching missed paraphrases**
3. **Edge Cases Detected** - Anomalies present
4. **Model Says "Not Paraphrase" but unsure** (0.40-0.75) - Might be wrong

**Agents stay dormant when**:
- HIGH confidence (> 0.85) and no edge cases
- Model is confident and reliable

This ensures agents **rescue paraphrases the model missed** while keeping most queries fast.unction and would be
considered equivalent in most contexts. PARAPHRASE."
```

### Agent 3: Paraphrase Analyzer (Final Decision) ⚖️
**Role**: Senior Paraphrase Detection Expert

**Expertise**:
- Synthesizes insights from other agents
- Makes final confident decision
- Provides clear reasoning

**Example Decision**:
```
DECISION: YES
CONFIDENCE: HIGH
REASONING: Despite length difference, both texts convey identical greeting
intent. Edge case (length mismatch) detected but semantically equivalent.
```──────────────────────────┐
│         CrewAI Multi-Agent System           │
├─────────────────────────────────────────────┤
│ Agent 1: Edge Case Specialist               │
│  - Analyzes anomalies & reliability         │
│  - Detects factors confusing the model      │
├─────────────────────────────────────────────┤
│ Agent 2: Semantic Validator                 │
│  - Deep semantic equivalence analysis       │
│  -6. Borderline Case ⚠️
**Description**: Similarity in uncertain range (0.55-0.70)

**Example**:
- Text 1: "What's your name?"
- Text 2: "May I know your name?"
- Model Score: 0.62

**Why it matters**: Model unsure - could be paraphrase or not

### 7. Low Confidence Paraphrase 🎯
**Description**: Very low similarity (< 0.55) but might still be paraphrase

**Example**:
- Text 1: "OK"
- Text 2: "That sounds good to me, I agree"
- Model Score: 0.42

**Why it matters**: **This is where agents rescue paraphrases!** Model says "not paraphrase" but agents can see they mean the same thing.ation          │
├─────────────────────────────────────────────┤
│ Agent 3: Paraphrase Analyzer (Final)        │
│  - Synthesizes agent insights               │
│  - Makes confident final decision           │
└─────────────────────────────────────────────┘
    ↓
Adjusted Response with Agent Reasoning
```

## Confidence Levels

The system classifies predictions into 4 confidence levels:

| Level | Similarity Range | Action |
|-------|------------------|--------|
| **HIGH** | > 0.85 | Trust model prediction directly |
| **MEDIUM** | 0.65 - 0.85 | Use model prediction, monitor for edge cases |
| **LOW** | 0.55 - 0.65 | Validate with edge case detection |
| **UNCERTAIN** | < 0.55 | Activate LLM validation |

## Edge Case Detection

The system automatically detects 5 types of edge cases:

### 1. Length Mismatch
**Description**: Texts with significantly different lengths (3x or more)

**Example**:
- Text 1: "Hi"
- Text 2: "Hello, how are you doing today?"

**Why it matters**: Short text vs long text comparisons can be unreliable

### 2. Short Text
**Description**: Both texts are very short (< 10 words)

**Example**:
- Text 1: "Good morning"
- Text 2: "Morning!"

**Why it matters**: Limited context makes similarity detection harder

### 3. Exact Match with Low Similarity
**Description**: Texts are identical but model gives low similarity

**Example**:
- Text 1: "Hello world"
- Text 2: "Hello world"
- Model Score: 0.45 (should be ~1.0)

**Why it matters**: Indicates potential model issue requiring LLM check

### 4. Numeric Heavy
**Description**: Texts contain > 30% numbers

**Example**:
- Text 1: "Order #12345 for $99.99"
- Text 2: "Purchase 12345 costs 99.99"

**Why it matters**: Numbers can dominate embeddings incorrectly

### 5. Special Characters Heavy
**Description**: Texts contain > 20% special characters

**Example**:
- Text 1: "Email: user@example.com!!!"
- Text 2: "Contact: user@example.com"

**Why it matters**: Special chars can skew semantic similarity

## CrewAI Multi-Agent Workflow

When edge cases or uncertain predictions are detected, the system uses **CrewAI's sequential process**:

### Sequential Agent Process

**Step 1: Edge Case Specialist analyzes anomalies**
```
Task: "Analyze these texts for edge cases that might confuse the ML model"
Output: "Length mismatch detected (3x ratio). Short text case. 
         Model reliability may be reduced."
```

**Step 2: Semantic Validator determines equivalence**
```
Task: "Determine if texts are semantically equivalent"
Output: "PARAPHRASE - Both express greeting intent despite length difference.
         Same core meaning and communicative function."
```

**Step 3: Paraphrase Analyzer makes final decision**
```
Task: "Synthesize insights and make final decision"
Output: 
DECISION: YES
CONFIDENCE: HIGH  
REASONING: Despite edge cases, semantic analysis confirms paraphrase.
           Length difference doesn't change core meaning.
```

### Agent Decision Blending

The system intelligently blends model and agent decisions:

| Model Score | Agent Decision | Agent Confidence | Final Decision | Similarity | Why |
|-------------|----------------|------------------|----------------|------------|-----|
| 0.42 | Paraphrase | HIGH | **Paraphrase** | 0.90 | Trust high-confidence agent |
| 0.65 | Paraphrase | MEDIUM | **Paraphrase** | 0.80 | Agent boosts borderline case |
| 0.45 | Not Paraphrase | MEDIUM | **Not Paraphrase** | 0.65 | Agent confirms model |
| 0.95 | Not Paraphrase | LOW | **Paraphrase** | 0.95 | Trust high-confidence model |
| 0.60 | Paraphrase | HIGH | **Paraphrase** | 0.90 | **Rescued paraphrase!** |

## API Integration

### Request with Agentic AI
```json
POST /inference/compare
{
  "text1": "What's your name?",
  "text2": "May I know your name?",
  "use_cache": true,
  "use_agent": true  // Enable agentic validation
}
```

### Response with Agent Metadata
```json
{
  "similarity": 0.87,
  "is_paraphrase": true,
  "inference_time_ms": 45.2,
  "cached": false,
  "agent_used": true,
  "confidence_level": "HIGH",
  "edge_cases": [],
  "llm_validation": false,
  "llm_reasoning": null,
  "original_similarity": 0.87,
  "adjusted_similarity": 0.87
}
```

### Response with CrewAI Agent Validation (Paraphrase Rescued!)
```json
{
  "similarity": 0.90,
  "is_paraphrase": true,
  "inference_time_ms": 3250.5,
  "cached": false,
  "agent_used": true,
  "confidence_level": "LOW",
  "edge_cases": ["short_text", "low_confidence_paraphrase"],
  "agent_validation": true,
  "agent_reasoning": "DECISION: YES\nCONFIDENCE: HIGH\nREASONING: Despite length difference and low model score, both texts express greeting intent. Semantically equivalent - same communicative function.",
  "agent_confidence": "HIGH",
  "paraphrase_rescued": true,
  "original_similarity": 0.45,
  "adjusted_similarity": 0.90
}
```

**🎯 Key Feature**CrewAI agent usage and paraphrase rescue rate:

```python
GET /inference/health
{
  "status": "healthy",
  "model_ready": true,
  "performance": {
    "total_requests": 1000,
    "agent_validations": 47,
    "agent_usage_rate": "4.70%",
    "agentic_ai": {
      "enabled": true,
      "total_validations": 1000,
      "agent_activations": 47,
      "activation_rate": "4.70%",
      "paraphrase_rescues": 12,
      "rescue_rate": "25.53%",
      "edge_cases_detected": {
        "length_mismatch": 8,
        "short_text": 15,
        "exact_match_low_similarity": 2,
        "numeric_heavy": 6,
        "special_chars_heavy": 4,
        "borderline_case": 18,
        "low_confidence_paraphrase": 12
      }
    }
  }
}
```

**Key Metrics**:
- **Activation Rate**: ~4-7% (agents only when needed)
- **Rescue Rate**: ~20-30% of activations catch paraphrases model missed
- **Latency**: +2-4 seconds when agents activated (acceptable for 4-7% of queries)   }
    }API Keys

**Gemini (Recommended - Free)**:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy key to `.env`: `GEMINI_API_KEY=your_key_here`

**Groq (Alternative - Free)**:
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up for free account
3. Create API key
4. Add to `.env`: `GROQ_API_KEY=your_key_here`

**Note**: CrewAI will use Gemini by default if both are provided. Groq uses LLaMA 3.3 70B.

### Environment Variables

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Get GCatches Paraphrases the Model Misses** 🎯
- Agents analyze borderline cases (0.55-0.75 similarity)
- Semantic validation beyond surface similarity
- **Rescue rate**: ~20-30% of agent activations correct model mistakes

### 2. **Multi-Agent Intelligence** 🤖
- 3 specialized agents with different expertise
- Collaborative analysis ensures comprehensive evaluation
- Sequential process builds on each agent's insights

### 3. **Smart Triggering** ⚡
- Only activates when needed (~4-7% of queries)
- HIGH confidence cases skip agents (fast path)
- Borderline and edge cases get expert review

### 4. **Improved Accuracy**
- Reduces false negatives (missed paraphrases)
- Handles edge cases that fool neural models
- Human-like semantic reasoning

### 5. **Transparency**
- Returns agent reasoning and confidence
- Shows which edge cases were detected
- Indicates when paraphrase was "rescued"

### 6. **Performance Balance**
- 93-96% of queries remain fast (<50ms)
- 4-7% get thorough agent review (+2-4s)
- Overall great UX with higher accuracy
  "text2": "...",
  "use_agent": false  // Disable agentic pipeline
}
```

## Benefits

### 1. **Improved Accuracy**
- Catches edge cases that fool neural models
- LLM provides human-like reasoning for ambiguous cases

### 2. **Transparency**
- Returns confidence levels and edge case flags
- Provides LLM reasoning when used

### 3. **Performance**
- Only activates LLM for ~1-5% of queries (edge cases)
- Most queries use fast model-only inference

### 4. **Robustness**
- Handles short texts, length mismatches, special characters
- Validates exact matches that score low

## Advanced Usage

### Custom Edge Case Thresholds

Modify [backend/services/agentic_validator.py](backend/services/agentic_validator.py):

```python
def check_edge_cases(self, text_a: str, text_b: str, similarity: float) -> List[str]:
    # Adjust thresholds
    LENGTH_MISMATCH_RATIO = 2.0  # Default: 3.0
    SHORT_TEXT_THRESHOLD = 15    # Default: 10
    NUMERIC_THRESHOLD = 0.2      # Default: 0.3
    # ... etc
```

### Custom Confidence Bands

Modify [backendParaphrase Rescued by Agents 🎯

**Scenario**: Model thinks texts are different, but they mean the same thing

```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "OK",
    "text2": "That sounds good to me, I agree with you",
    "use_agent": true
  }'
```, `borderline_case`
- CrewAI validates: Both are greetings → Paraphrase
- Confidence: HIGH
- Adjusted similarity: 0.88

---

### Example 3: Length Mismatch - Model Correct ❌

**Agent Analysis**:
- Edge Cases: ["length_mismatch", "short_text", "low_confidence_paraphrase"]
- Semantic Validator: "Both express agreement - semantically equivalent"
- Final Decision: **Paraphrase** ✅
- Confidence: HIGH
- **paraphrase_rescued: true**

**Result**: Agents caught what the model missed!

---

### Example 2: /services/agentic_validator.py](backend/services/agentic_validator.py):
CrewAI validates: Different meanings → Not Paraphrase
- Confidence: HIGH
- Agents confirm model decision

---

### Example 4: Numeric Heavy - Paraphrase Detected More strict HIGH threshold
        return ConfidenceLevel.HIGH
    elif similarity > 0.70:
        return ConfidenceLevel.MEDIUM
    # ... etc
```

## Troubleshooting

### Issue: LLM not activating
CrewAI validates: Same transaction → Paraphrase
- Confidence: HIGH
- Adjusted similarity: 0.92

---

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Agentic AI Design Patterns](https://www.anthropic.com/research/building-effective-agents)
- [Multi-Agent Systems in NLP](https://arxiv.org/abs/2308.08155
**Solution**:
- LLM calls add ~1-2s latency
- Only affects ~1-5% of queries (edge cases)
- Use `use_agent=false` for speed-critical applications

### Issue: LLM validation errors

**Check**:
1. Valid Gemini API key
2. Internet connection
3. Check logs for error details

## Examples

### Example 1: Short Text Edge Case
```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Hi",
    "text2": "Hello!",
    "use_agent": true
  }'
```

**Response**:
- Edge case: `short_text`
- LLM validates: Both are greetings → Paraphrase
- Confidence: HIGH

### Example 2: Length Mismatch
```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "OK",
    "text2": "That sounds good to me, I agree with your proposal",
    "use_agent": true
  }'
```

**Response**:
- Edge case: `length_mismatch`
- LLM validates: Different semantic meanings → Not Paraphrase
- Confidence: HIGH

### Example 3: Numeric Heavy
```bash
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Order 12345 for $99.99",
    "text2": "Purchase #12345 costs 99.99",
    "use_agent": true
  }'
```

**Response**:
- Edge case: `numeric_heavy`
- LLM validates: Same transaction → Paraphrase
- Confidence: HIGH

## References

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Agentic AI Design Patterns](https://www.anthropic.com/research/building-effective-agents)
- [Edge Case Handling in NLP](https://arxiv.org/abs/2104.08821)
