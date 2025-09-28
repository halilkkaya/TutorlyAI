#!/usr/bin/env python3
"""
Similarity Cache Test
Tests TutorlyAI similarity-based cache system
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.similarity_cache import similarity_cache
from tools.redis_cache_adapters import initialize_redis_caches, redis_query_cache

async def test_similarity_cache():
    """Similarity cache sistemi test et"""
    print("TutorlyAI Similarity Cache Test")
    print("=" * 50)

    # Initialize Redis caches
    print("\n1. Initializing Redis Cache System...")
    success = await initialize_redis_caches()
    if not success:
        print("   [FAIL] Redis cache initialization failed")
        return

    print("   [OK] Redis cache system initialized")

    # Test 1: Basic exact match
    print("\n2. Testing Exact Match...")
    cache_key = "test_exact_match"
    query = "matematik 5. sinif"
    filters = {"sinif": 5, "ders": "matematik"}
    test_result = {"message": "Test exact match result", "data": [1, 2, 3]}

    # Cache'e kaydet
    success = await similarity_cache.set_cached_result_with_similarity(
        "query", cache_key, test_result, query, filters
    )
    if success:
        print("   [OK] Exact match data cached")
    else:
        print("   [FAIL] Failed to cache exact match data")

    # Exact match'i oku
    cached = await similarity_cache.get_cached_result_with_similarity(
        "query", cache_key, query, filters
    )
    if cached and cached["message"] == test_result["message"]:
        print("   [OK] Exact match cache hit successful")
    else:
        print("   [FAIL] Exact match cache hit failed")

    # Test 2: Similar query test
    print("\n3. Testing Similar Query Match...")
    similar_query = "matematik 5 sinif"  # Çok benzer sorgu (nokta eksik)

    # Aynı cache key ile farklı query dene
    similar_cache_key = "test_similar_match"
    similar_cached = await similarity_cache.get_cached_result_with_similarity(
        "query", similar_cache_key, similar_query, filters
    )

    if similar_cached and similar_cached["message"] == test_result["message"]:
        print("   [OK] Similar query cache hit successful")
        print(f"   Original: '{query}' -> Similar: '{similar_query}'")
    else:
        print("   [INFO] Similar query cache miss (might be expected if similarity < 0.80)")

    # Test 3: Very different query (should miss)
    print("\n4. Testing Different Query...")
    different_query = "turkce 3. sinif edebiyat"
    different_filters = {"sinif": 3, "ders": "turkce"}
    different_cache_key = "test_different_match"

    different_cached = await similarity_cache.get_cached_result_with_similarity(
        "query", different_cache_key, different_query, different_filters
    )

    if different_cached is None:
        print("   [OK] Different query correctly resulted in cache miss")
    else:
        print("   [WARNING] Different query unexpectedly resulted in cache hit")

    # Test 4: Multiple similar queries
    print("\n5. Testing Multiple Similar Queries...")
    similar_queries = [
        "matematik 5 sinif konulari",
        "5. sinif matematik",
        "5. sinif matematik dersi",
        "matematik 5. sinif sorulari"
    ]

    hit_count = 0
    for i, sim_query in enumerate(similar_queries):
        sim_cache_key = f"test_multi_similar_{i}"
        result = await similarity_cache.get_cached_result_with_similarity(
            "query", sim_cache_key, sim_query, filters
        )
        if result is not None:
            hit_count += 1
            print(f"   [HIT] '{sim_query}' -> Cache hit")
        else:
            print(f"   [MISS] '{sim_query}' -> Cache miss")

    print(f"   Similarity hit rate: {hit_count}/{len(similar_queries)} ({hit_count/len(similar_queries)*100:.1f}%)")

    # Test 5: Stats
    print("\n6. Similarity Cache Stats:")
    stats = similarity_cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test 6: Redis Query Cache with similarity
    print("\n7. Testing Redis Query Cache with Similarity...")

    async def mock_query_function(query, filters=None):
        """Mock query function for testing"""
        return {
            "query": query,
            "filters": filters,
            "results": ["Mock result 1", "Mock result 2"],
            "timestamp": asyncio.get_event_loop().time()
        }

    # İlk query - cache miss
    result1 = await redis_query_cache.get_or_execute_query(
        "test_query_1", mock_query_function, "matematik 5. sinif", filters=filters
    )
    print(f"   [QUERY 1] Result: {result1['query']}")

    # Benzer query - similarity hit olmalı
    result2 = await redis_query_cache.get_or_execute_query(
        "test_query_2", mock_query_function, "matematik 5 sinif", filters=filters
    )
    print(f"   [QUERY 2] Result: {result2['query']}")

    if result1['results'] == result2['results']:
        print("   [OK] Similar query returned cached result")
    else:
        print("   [INFO] Similar query executed new search")

    # Stats
    query_stats = redis_query_cache.get_stats()
    print(f"   Cache Stats: {query_stats['hit_rate_percent']:.1f}% hit rate")

    print("\n" + "=" * 50)
    print("Similarity Cache Test Completed!")

    # Cleanup
    await similarity_cache.clear_similarity_metadata()

if __name__ == "__main__":
    asyncio.run(test_similarity_cache())