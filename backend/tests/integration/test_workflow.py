#!/usr/bin/env python3
"""
Test script to validate the distillation and fine-tuning workflow improvements.
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"  # Adjust if needed

def test_workflow_templates():
    """Test workflow templates API."""
    print("🧪 Testing Workflow Templates API...")
    
    try:
        # Test list templates
        response = requests.get(f"{BASE_URL}/api/v1/finetune/workflow-templates")
        if response.status_code == 200:
            data = response.json()
            templates = data.get("templates", [])
            print(f"✅ Found {len(templates)} workflow templates")
            for template in templates:
                print(f"   - {template['name']} ({template['difficulty']})")
            return templates
        else:
            print(f"❌ Failed to fetch templates: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error testing templates: {e}")
        return []

def test_distillation_job_creation():
    """Test creating a distillation job with dataset creation enabled."""
    print("\n🧪 Testing Distillation Job Creation...")
    
    job_data = {
        "name": "Test Coding Assistant",
        "description": "Test job for workflow validation",
        "config": {
            "teacher_provider": "openai",
            "teacher_model": "gpt-4o-mini",  # Cheaper for testing
            "temperature": 0.3,
            "max_tokens": 1024,
            "target_examples": 5,  # Small for quick testing
            "create_managed_dataset": True,
            "dataset_name": "Test Coding Dataset",
            "dataset_description": "Small test dataset for validation"
        },
        "prompts": [
            "Write a Python function to reverse a string",
            "Create a simple calculator in JavaScript",
            "Explain how to use list comprehensions",
            "Debug this code: print('hello world')",
            "Write a function to find the maximum value in a list"
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/finetune/distillation/jobs",
            json=job_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            job = response.json()
            print(f"✅ Created distillation job: {job['id']}")
            print(f"   Name: {job['name']}")
            print(f"   Dataset creation: {job['config']['create_managed_dataset']}")
            return job
        else:
            print(f"❌ Failed to create job: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating job: {e}")
        return None

def test_job_status(job_id: str):
    """Test job status and progress tracking."""
    print(f"\n🧪 Testing Job Status Tracking for {job_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/finetune/distillation/jobs/{job_id}")
        if response.status_code == 200:
            job = response.json()
            print(f"✅ Job Status: {job['status']}")
            print(f"   Progress: {job['progress']:.1f}%")
            print(f"   Generated: {job['metrics']['generated_count']}")
            print(f"   Dataset ID: {job.get('output_dataset_id', 'Not created yet')}")
            return job
        else:
            print(f"❌ Failed to get job status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting job status: {e}")
        return None

def test_quality_assessment(job_id: str):
    """Test quality assessment for completed jobs."""
    print(f"\n🧪 Testing Quality Assessment for {job_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/finetune/distillation/jobs/{job_id}/quality-assessment")
        if response.status_code == 200:
            assessment = response.json()
            print(f"✅ Quality Grade: {assessment['overall_grade']}")
            print(f"   Success Rate: {assessment['metrics']['success_rate']}%")
            print(f"   Avg Quality: {assessment['metrics']['avg_quality_score']}")
            print(f"   Ready for Training: {assessment['ready_for_training']}")
            print(f"   Recommendations: {len(assessment['recommendations'])}")
            return assessment
        else:
            print(f"❌ Failed to get quality assessment: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting quality assessment: {e}")
        return None

def test_training_job_creation(job_id: str):
    """Test automatic training job creation from distillation."""
    print(f"\n🧪 Testing Training Job Creation from {job_id}...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/finetune/distillation/jobs/{job_id}/create-training-job")
        if response.status_code == 200:
            data = response.json()
            training_job = data['training_job']
            print(f"✅ Created training job: {training_job['id']}")
            print(f"   Name: {training_job['name']}")
            print(f"   Dataset: {training_job['dataset_id']}")
            print(f"   Base Model: {training_job['base_model']}")
            print(f"   Recommendations applied: {len(data['recommendations'])} settings")
            return training_job
        else:
            print(f"❌ Failed to create training job: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating training job: {e}")
        return None

def main():
    """Run all workflow tests."""
    print("🚀 Testing Llama Nexus Distillation → Fine-Tuning Workflow")
    print("=" * 60)
    
    # Test 1: Workflow Templates
    templates = test_workflow_templates()
    
    # Test 2: Create Distillation Job
    job = test_distillation_job_creation()
    if not job:
        print("\n❌ Cannot continue tests without a valid job")
        return
    
    job_id = job['id']
    
    # Test 3: Monitor Job (basic status check)
    test_job_status(job_id)
    
    print(f"\n📝 Manual Testing Required:")
    print(f"   1. Visit: http://localhost:3000/finetuning/templates")
    print(f"   2. Check job progress at: http://localhost:3000/finetuning/distillation")
    print(f"   3. Wait for job {job_id} to complete")
    print(f"   4. Test quality assessment and training job creation")
    
    print(f"\n🔗 Useful URLs for Testing:")
    print(f"   - Workflow Templates: http://localhost:3000/finetuning/templates")
    print(f"   - Distillation Jobs: http://localhost:3000/finetuning/distillation")
    print(f"   - Fine-Tuning: http://localhost:3000/finetuning")
    print(f"   - API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()