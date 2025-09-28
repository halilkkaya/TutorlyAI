#!/bin/sh
# TutorlyAI Redis Health Check Script

# Redis'in çalışıp çalışmadığını kontrol et
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Redis is not responding to ping"
    exit 1
fi

# Memory kullanımını kontrol et
MEMORY_USAGE=$(redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
if [ -z "$MEMORY_USAGE" ]; then
    echo "Cannot get memory usage information"
    exit 1
fi

# TutorlyAI cache database'lerinin erişilebilir olduğunu kontrol et
# Database 0 (Query Cache)
redis-cli -n 0 ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Query cache database (DB 0) is not accessible"
    exit 1
fi

# Database 1 (Performance Cache)
redis-cli -n 1 ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Performance cache database (DB 1) is not accessible"
    exit 1
fi

# Database 2 (BM25 Cache)
redis-cli -n 2 ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "BM25 cache database (DB 2) is not accessible"
    exit 1
fi

echo "TutorlyAI Redis Cache is healthy - Memory: $MEMORY_USAGE"
exit 0