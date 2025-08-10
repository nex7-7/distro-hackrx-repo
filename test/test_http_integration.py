"""
Test Script for HTTP Tools Integration

This script demonstrates how the agentic riddle solver can now make HTTP requests
during its reasoning process. Run this to test the integration.
"""
import asyncio
import httpx
from components.agentic_solver import solve_riddle_with_query


async def test_http_integration():
    """Test the HTTP tools integration with a sample query."""

    # Sample query that might require HTTP requests
    test_query = "What is the current status of the HackRx API endpoints? Check the health of the main API."

    async with httpx.AsyncClient() as client:
        try:
            print("🧪 Testing HTTP tools integration...")
            print(f"📝 Query: {test_query}")
            print("\n" + "="*60)

            # This will trigger the agentic system
            result = await solve_riddle_with_query(client, test_query)

            print("\n" + "="*60)
            print("🎯 Result:")
            print(result)

        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting HTTP Tools Integration Test")
    asyncio.run(test_http_integration())
