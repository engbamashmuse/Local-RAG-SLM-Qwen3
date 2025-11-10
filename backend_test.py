import requests
import sys
import json
from datetime import datetime
import tempfile
import os
from pathlib import Path

class RAGSLMAPITester:
    def __init__(self, base_url="https://mini-document-rag.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details="", expected_status=None, actual_status=None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
            if expected_status and actual_status:
                print(f"   Expected status: {expected_status}, Got: {actual_status}")
        
        self.test_results.append({
            "test_name": name,
            "status": "PASSED" if success else "FAILED",
            "details": details,
            "expected_status": expected_status,
            "actual_status": actual_status
        })

    def test_root_endpoint(self):
        """Test root API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Response: {response.json()}" if success else f"Error: {response.text}"
            self.log_test("Root API Endpoint", success, details, 200, response.status_code)
            return success
        except Exception as e:
            self.log_test("Root API Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_health_endpoint(self):
        """Test health check endpoint - should show Ollama disconnected"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                # Check if Ollama is properly reported as disconnected
                ollama_status = data.get('ollama', 'unknown')
                if ollama_status == 'disconnected':
                    details = f"Health check working correctly - Ollama disconnected as expected: {data}"
                else:
                    success = False
                    details = f"Unexpected Ollama status: {ollama_status}. Expected 'disconnected'"
            else:
                details = f"Health endpoint failed: {response.text}"
            
            self.log_test("Health Check Endpoint", success, details, 200, response.status_code)
            return success, response.json() if success else {}
        except Exception as e:
            self.log_test("Health Check Endpoint", False, f"Exception: {str(e)}")
            return False, {}

    def test_documents_list_endpoint(self):
        """Test documents list endpoint"""
        try:
            response = requests.get(f"{self.api_url}/documents", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                documents = data.get('documents', [])
                count = data.get('count', 0)
                details = f"Found {count} documents"
            else:
                details = f"Error: {response.text}"
            
            self.log_test("Documents List Endpoint", success, details, 200, response.status_code)
            return success
        except Exception as e:
            self.log_test("Documents List Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_collections_endpoint(self):
        """Test collections endpoint"""
        try:
            response = requests.get(f"{self.api_url}/collections", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                collections = data.get('collections', [])
                details = f"Found collections: {collections}"
            else:
                details = f"Error: {response.text}"
            
            self.log_test("Collections Endpoint", success, details, 200, response.status_code)
            return success
        except Exception as e:
            self.log_test("Collections Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_upload_endpoint_without_ollama(self):
        """Test upload endpoint - should return 503 when Ollama is not running"""
        try:
            # Create a simple test file
            test_content = "This is a test document for RAG-SLM testing."
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file_path = f.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    files = {'files': ('test_document.txt', f, 'text/plain')}
                    data = {'collection': 'test_collection'}
                    
                    response = requests.post(
                        f"{self.api_url}/documents/upload",
                        files=files,
                        data=data,
                        timeout=15
                    )
                
                # Should return 503 when Ollama is not running
                expected_status = 503
                success = response.status_code == expected_status
                
                if success:
                    details = "Upload correctly returns 503 when Ollama is not running"
                else:
                    details = f"Upload should return 503 when Ollama is not running, got {response.status_code}: {response.text}"
                
                self.log_test("Upload Endpoint (No Ollama)", success, details, expected_status, response.status_code)
                return success
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            self.log_test("Upload Endpoint (No Ollama)", False, f"Exception: {str(e)}")
            return False

    def test_chat_endpoint_without_ollama(self):
        """Test chat endpoint - should return 503 when Ollama is not running"""
        try:
            chat_data = {
                "query": "What is this document about?",
                "collection": "test_collection",
                "session_id": "test_session"
            }
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=chat_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            # Should return 503 when Ollama is not running
            expected_status = 503
            success = response.status_code == expected_status
            
            if success:
                details = "Chat correctly returns 503 when Ollama is not running"
            else:
                details = f"Chat should return 503 when Ollama is not running, got {response.status_code}: {response.text}"
            
            self.log_test("Chat Endpoint (No Ollama)", success, details, expected_status, response.status_code)
            return success
            
        except Exception as e:
            self.log_test("Chat Endpoint (No Ollama)", False, f"Exception: {str(e)}")
            return False

    def test_invalid_endpoints(self):
        """Test invalid endpoints return proper 404"""
        invalid_endpoints = [
            "/api/invalid",
            "/api/documents/invalid",
            "/api/chat/invalid"
        ]
        
        all_passed = True
        for endpoint in invalid_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                success = response.status_code == 404
                if not success:
                    all_passed = False
                details = f"Endpoint {endpoint} returned {response.status_code}"
                self.log_test(f"Invalid Endpoint {endpoint}", success, details, 404, response.status_code)
            except Exception as e:
                all_passed = False
                self.log_test(f"Invalid Endpoint {endpoint}", False, f"Exception: {str(e)}")
        
        return all_passed

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🚀 Starting RAG-SLM Backend API Tests")
        print("=" * 50)
        
        # Test basic connectivity
        print("\n📡 Testing Basic Connectivity...")
        self.test_root_endpoint()
        
        # Test health check
        print("\n🏥 Testing Health Check...")
        health_success, health_data = self.test_health_endpoint()
        
        # Test document endpoints
        print("\n📄 Testing Document Endpoints...")
        self.test_documents_list_endpoint()
        self.test_collections_endpoint()
        
        # Test upload endpoint (should fail gracefully without Ollama)
        print("\n📤 Testing Upload Endpoint (Expected to fail without Ollama)...")
        self.test_upload_endpoint_without_ollama()
        
        # Test chat endpoint (should fail gracefully without Ollama)
        print("\n💬 Testing Chat Endpoint (Expected to fail without Ollama)...")
        self.test_chat_endpoint_without_ollama()
        
        # Test invalid endpoints
        print("\n❌ Testing Invalid Endpoints...")
        self.test_invalid_endpoints()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if health_success:
            print(f"\n🏥 Health Status: {health_data}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test function"""
    tester = RAGSLMAPITester()
    
    try:
        success = tester.run_all_tests()
        
        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": tester.tests_run,
            "passed_tests": tester.tests_passed,
            "failed_tests": tester.tests_run - tester.tests_passed,
            "success_rate": (tester.tests_passed/tester.tests_run)*100 if tester.tests_run > 0 else 0,
            "test_details": tester.test_results
        }
        
        # Write results to file
        with open('/app/backend_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📝 Detailed results saved to: /app/backend_test_results.json")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())