# TutorlyAI Redis Cache Container
FROM redis:7.2-alpine

# Redis kullanıcısı zaten mevcut - kontrol et
RUN id redis || (addgroup -g 999 redis && adduser -D -s /bin/sh -u 999 -G redis redis)

# TutorlyAI Cache için özel konfigürasyon
LABEL maintainer="TutorlyAI Team"
LABEL description="Redis cache container optimized for TutorlyAI RAG system"
LABEL version="1.0.0"

# Gerekli paketleri yükle
RUN apk add --no-cache \
    tzdata \
    ca-certificates

# Timezone ayarla
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Redis konfigürasyon dosyasını kopyala
COPY conf/redis.conf /usr/local/etc/redis/redis.conf

# Log klasörü oluştur
RUN mkdir -p /var/log/redis && chown redis:redis /var/log/redis

# Data klasörü oluştur
RUN mkdir -p /data && chown redis:redis /data

# Redis kullanıcısına geç
USER redis

# Health check script
COPY --chown=redis:redis healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Redis portunu aç
EXPOSE 6379

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
  CMD /usr/local/bin/healthcheck.sh

# Redis'i başlat
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]