#!/usr/bin/env python3
"""
Redis Cache Integration Test
Tests TutorlyAI Redis cache system
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.redis_client import redis_client, initialize_redis, initialize_redis_async

async def test_redis_integration():
    """Redis entegrasyonu test et"""
    print("TutorlyAI Redis Cache Integration Test")
    print("=" * 50)

    # 1. Sync Redis test
    print("\n1. Testing Sync Redis Connection...")
    try:
        success = initialize_redis()
        if success:
            print("   ✅ Sync Redis initialized successfully")

            # Test sync operations
            test_key = "test:sync:key"
            test_value = {"message": "Hello from sync Redis!", "timestamp": "2025-09-28"}

            # Set cache
            result = redis_client.set_cache("performance", test_key, test_value, 60)
            if result:
                print("   [OK] Sync cache set successful")
            else:
                print("   [FAIL] Sync cache set failed")

            # Get cache
            retrieved = redis_client.get_cache("performance", test_key)
            if retrieved and retrieved["message"] == test_value["message"]:
                print("   [OK] Sync cache get successful")
                print(f"   Retrieved: {retrieved}")
            else:
                print("   [FAIL] Sync cache get failed")

            # Delete cache
            deleted = redis_client.delete_cache("performance", test_key)
            if deleted:
                print("   [OK] Sync cache delete successful")
            else:
                print("   [FAIL] Sync cache delete failed")

        else:
            print("   [FAIL] Sync Redis initialization failed")

    except Exception as e:
        print(f"   [FAIL] Sync Redis error: {str(e)}")

    # 2. Async Redis test
    print("\n2. Testing Async Redis Connection...")
    try:
        success = await initialize_redis_async()
        if success:
            print("   [OK] Async Redis initialized successfully")

            # Test async operations
            test_key = "test:async:key"
            test_value = {"message": "Hello from async Redis!", "data": [1, 2, 3, 4, 5]}

            # Set cache async
            result = await redis_client.set_cache_async("query", test_key, test_value, 120)
            if result:
                print("   [OK] Async cache set successful")
            else:
                print("   [FAIL] Async cache set failed")

            # Get cache async
            retrieved = await redis_client.get_cache_async("query", test_key)
            if retrieved and retrieved["message"] == test_value["message"]:
                print("   [OK] Async cache get successful")
                print(f"   Retrieved: {retrieved}")
            else:
                print("   [FAIL] Async cache get failed")

            # Delete cache async
            deleted = await redis_client.delete_cache_async("query", test_key)
            if deleted:
                print("   [OK] Async cache delete successful")
            else:
                print("   [FAIL] Async cache delete failed")

        else:
            print("   [FAIL] Async Redis initialization failed")

    except Exception as e:
        print(f"   [FAIL] Async Redis error: {str(e)}")

    # 3. Health check test
    print("\n3. Testing Redis Health Check...")
    try:
        health = await redis_client.health_check()
        print(f"   Health Status: {health}")

        if health["redis_available"]:
            print("   [OK] Redis health check passed")

            # Database statistics
            for db_name, db_status in health["databases"].items():
                if db_status.get("available"):
                    print(f"   [OK] Database '{db_name}': Available")
                else:
                    print(f"   [FAIL] Database '{db_name}': Not available")
        else:
            print("   [FAIL] Redis health check failed")

    except Exception as e:
        print(f"   [FAIL] Health check error: {str(e)}")

    # 4. Multiple database test
    print("\n4. Testing Multiple Database Access...")
    try:
        databases = ["query", "performance", "bm25", "session"]

        for db_name in databases:
            test_key = f"test:db:{db_name}"
            test_value = f"Database {db_name} test data"

            # Test each database
            await redis_client.set_cache_async(db_name, test_key, test_value, 30)
            retrieved = await redis_client.get_cache_async(db_name, test_key)

            if retrieved == test_value:
                print(f"   [OK] Database '{db_name}': Working correctly")
            else:
                print(f"   [FAIL] Database '{db_name}': Failed")

    except Exception as e:
        print(f"   [FAIL] Multiple database test error: {str(e)}")

    # 5. Cache key generation test
    print("\n5. Testing Cache Key Generation...")
    try:
        # Test string data
        key1 = redis_client.generate_cache_key("test", "simple_string")
        print(f"   String key: {key1}")

        # Test dict data
        key2 = redis_client.generate_cache_key("test", {"query": "test", "filters": {"class": 5}})
        print(f"   Dict key: {key2}")

        # Test list data
        key3 = redis_client.generate_cache_key("test", ["item1", "item2", "item3"])
        print(f"   List key: {key3}")

        print("   [OK] Cache key generation working")

    except Exception as e:
        print(f"   [FAIL] Cache key generation error: {str(e)}")

    print("\n" + "=" * 50)
    print("Redis Integration Test Completed!")

    # Cleanup
    await redis_client.cleanup()

if __name__ == "__main__":
    asyncio.run(test_redis_integration())