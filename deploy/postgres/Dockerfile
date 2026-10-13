# Postgres 18 with pgvector, plus pg_textsearch: BM25 ranking inside Postgres
# (Tiger Data, PostgreSQL licence). The release file is pinned by its SHA-256, so a
# changed download fails the build. Debian 13 (trixie): the prebuilt library needs
# glibc 2.38 or later, and Debian 12 has 2.36.
FROM pgvector/pgvector:0.8.6-pg18-trixie

ADD --checksum=sha256:ce65e92b622440e02985e2f4a97f9035efc4788240418b08d147c0a3f6e7d2f3 \
    https://github.com/timescale/pg_textsearch/releases/download/v1.4.0/pg_textsearch-pg18-v1.4.0.tar.gz \
    /tmp/pg_textsearch.tar.gz

RUN tar -xzf /tmp/pg_textsearch.tar.gz -C /tmp \
    && cp /tmp/pg18/pg_textsearch.so /usr/lib/postgresql/18/lib/ \
    && cp /tmp/pg18/pg_textsearch.control /tmp/pg18/pg_textsearch--*.sql \
        /usr/share/postgresql/18/extension/ \
    && rm -rf /tmp/pg18 /tmp/pg_textsearch.tar.gz
