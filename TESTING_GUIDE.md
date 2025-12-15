# 🚀 Distillation → Fine-Tuning Workflow Testing Guide

## System Readiness Status

✅ **Backend Components Ready:**
- Workflow templates with 5 pre-configured scenarios
- Smart hyperparameter recommendations API
- One-click training job creation endpoint
- Quality assessment and grading system
- Enhanced distillation manager with dataset integration

✅ **Frontend Components Ready:**
- Workflow Templates page (`/finetuning/templates`)
- Enhanced Distillation page with progress tracking
- One-click training buttons
- Quality assessment visualization
- End-to-end workflow progress indicators

✅ **API Endpoints Available:**
- `GET /api/v1/finetune/workflow-templates` - List templates
- `POST /api/v1/finetune/workflow-templates/{id}/start` - Start workflow
- `POST /api/v1/finetune/distillation/jobs/{id}/create-training-job` - Auto-create training
- `GET /api/v1/finetune/distillation/jobs/{id}/quality-assessment` - Quality analysis

## Prerequisites for Testing

### 1. Environment Setup
```bash
# Ensure services are running
cd /home/alec/git/llama-nexus
docker-compose up -d

# Check services
docker-compose ps
```

### 2. API Keys Required
- **OpenAI API Key** (for GPT-4o teacher model)
- **Anthropic API Key** (optional, for Claude teacher)

### 3. Quick Validation
```bash
# Run the automated test script
python test_workflow.py
```

---

## 🧪 Test Workflow 1: Beginner (Recommended First Test)

**Objective:** Test the complete workflow using a pre-built template

### Step 1: Access Workflow Templates
1. Navigate to: `http://localhost:3000/finetuning/templates`
2. Verify 5 templates are displayed:
   - 🔧 Coding Assistant (Beginner)
   - ✍️ Creative Writing Assistant (Beginner)  
   - 🧠 Reasoning & Math Tutor (Intermediate)
   - 🎯 Domain Expert (Advanced)
   - 💬 Conversational AI (Intermediate)

### Step 2: Start Coding Assistant Workflow
1. Click on "🔧 Coding Assistant" template
2. Review the configuration:
   - Teacher: GPT-4o
   - Strategy: Chain-of-Thought
   - Examples: 200
   - Sample prompts displayed
3. **Option A:** Click "🚀 Start with Sample Prompts"
4. **Option B:** Add custom prompts and click "🎯 Start with Custom Prompts"

### Step 3: Monitor Distillation Progress
1. Should auto-redirect to `/finetuning/distillation`
2. Verify workflow progress tracker shows:
   - Step 1: Distillation (in progress)
   - Step 2: Dataset Creation (pending)
   - Step 3: Fine-Tuning (pending)
   - Step 4: Deploy Model (pending)
3. Watch real-time progress bar and example count

### Step 4: Quality Assessment (After Completion)
1. Wait for distillation to complete (~15-30 minutes for 200 examples)
2. Verify quality assessment appears:
   - Overall grade (A-D)
   - Success rate percentage
   - Quality metrics
   - Specific recommendations
3. Check "Dataset Created" indicator appears

### Step 5: One-Click Training
1. Click "🚀 Start Training" button
2. Verify automatic redirect to fine-tuning page
3. Confirm training job created with smart hyperparameters

**Expected Results:**
- ✅ Complete workflow in 5 clicks
- ✅ Automatic dataset creation
- ✅ Quality assessment with actionable feedback
- ✅ Smart hyperparameter selection
- ✅ Seamless transition to training

---

## 🔬 Test Workflow 2: Advanced (Power User Test)

**Objective:** Test advanced features and customization options

### Step 1: Custom Distillation Job
1. Navigate to: `http://localhost:3000/finetuning/distillation`
2. Create custom job with these settings:
   ```
   Name: Advanced Reasoning Test
   Teacher: GPT-4o or o1-mini
   Strategy: Thinking Tokens
   Quality: Strict (0.85 threshold)
   Self-Consistency: Enabled (3 samples)
   Response Refinement: Enabled (2 attempts)
   Dataset Creation: Enabled
   Dataset Name: Advanced Reasoning Dataset
   ```

### Step 2: Advanced Prompt Engineering
Add these challenging prompts:
```
Solve this step-by-step: A train leaves Chicago at 2 PM traveling 60 mph. Another train leaves New York (800 miles away) at 3 PM traveling 80 mph toward Chicago. When do they meet?

Explain the philosophical implications of Gödel's incompleteness theorems for artificial intelligence

Design a distributed system architecture for handling 1 million concurrent users with 99.99% uptime

Debug this Python code and explain the issue: def factorial(n): return n * factorial(n-1)

Write a proof by induction that the sum of first n odd numbers equals n²
```

### Step 3: Monitor Advanced Features
1. Watch for quality validation in real-time
2. Observe response refinement attempts
3. Check self-consistency voting results
4. Monitor thinking token processing

### Step 4: Quality Analysis Deep Dive
1. After completion, analyze quality assessment:
   - Grade breakdown by quality dimensions
   - Refinement rate analysis
   - Consistency agreement scores
   - Detailed recommendations
2. Compare metrics across different strategies

### Step 5: Custom Training Configuration
1. Use "🚀 Start Training" but modify hyperparameters:
   - Adjust learning rate based on recommendations
   - Modify batch size for your hardware
   - Set custom evaluation metrics
2. Test training job creation API directly:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/finetune/distillation/jobs/{job_id}/create-training-job"
   ```

### Step 6: Multi-Template Comparison
1. Start 2-3 different templates simultaneously
2. Compare quality assessments
3. Analyze which strategies work best for different use cases
4. Test template customization options

**Expected Advanced Results:**
- ✅ High-quality outputs with thinking tokens
- ✅ Sophisticated quality validation
- ✅ Detailed performance analytics
- ✅ Customizable hyperparameter recommendations
- ✅ Multi-strategy comparison capabilities

---

## 🐛 Troubleshooting Common Issues

### Issue 1: Templates Not Loading
```bash
# Check API endpoint
curl http://localhost:8000/api/v1/finetune/workflow-templates

# Check backend logs
docker-compose logs backend
```

### Issue 2: Distillation Job Fails
- Verify API keys are set correctly
- Check teacher model availability
- Review prompt complexity (start simple)
- Monitor rate limiting

### Issue 3: Dataset Not Created
- Ensure `create_managed_dataset: true` in config
- Check dataset store initialization
- Verify file permissions in data directory

### Issue 4: Quality Assessment Missing
- Only available for completed jobs
- Requires generated examples > 0
- Check API endpoint accessibility

### Issue 5: Training Job Creation Fails
- Verify distillation job completed successfully
- Check dataset was created (`output_dataset_id` exists)
- Ensure training manager is initialized

---

## 📊 Success Metrics to Validate

### Functional Metrics
- [ ] All 5 workflow templates load correctly
- [ ] Distillation jobs complete successfully
- [ ] Datasets are automatically created
- [ ] Quality assessments generate grades A-D
- [ ] Training jobs create with smart hyperparameters
- [ ] One-click workflow completes end-to-end

### Performance Metrics  
- [ ] Template loading < 2 seconds
- [ ] Job creation < 5 seconds
- [ ] Quality assessment < 10 seconds
- [ ] Training job creation < 3 seconds

### User Experience Metrics
- [ ] Workflow progress clearly visible
- [ ] Error messages are actionable
- [ ] Recommendations are specific and helpful
- [ ] Navigation between pages is seamless
- [ ] Status updates are real-time

---

## 🎯 Test Scenarios by Use Case

### Scenario A: Data Scientist (New to LLMs)
1. Start with "🔧 Coding Assistant" template
2. Use default settings throughout
3. Follow guided workflow
4. Expect: Easy onboarding, clear guidance

### Scenario B: ML Engineer (Experienced)
1. Use "🧠 Reasoning & Math Tutor" template
2. Customize prompts and hyperparameters
3. Analyze quality metrics deeply
4. Expect: Advanced controls, detailed analytics

### Scenario C: Researcher (Experimental)
1. Create custom distillation job
2. Test multiple strategies
3. Compare quality assessments
4. Expect: Flexibility, comprehensive metrics

### Scenario D: Production Team (Efficiency Focus)
1. Use workflow templates for speed
2. Rely on automated recommendations
3. Focus on end-to-end automation
4. Expect: Minimal manual intervention

---

## 🔗 Quick Links for Testing

- **Workflow Templates:** http://localhost:3000/finetuning/templates
- **Distillation Jobs:** http://localhost:3000/finetuning/distillation  
- **Fine-Tuning Overview:** http://localhost:3000/finetuning
- **API Documentation:** http://localhost:8000/docs
- **Backend Health:** http://localhost:8000/health

## 📝 Feedback Collection

During testing, note:
1. **Pain Points:** Where did you get stuck?
2. **Confusion:** What wasn't clear?
3. **Missing Features:** What would make it better?
4. **Performance:** What felt slow?
5. **Success Stories:** What worked really well?

This feedback will help prioritize future improvements to the workflow experience.