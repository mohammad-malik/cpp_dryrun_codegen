FROM alpine:3.19
RUN apk add --no-cache g++ build-base
WORKDIR /code
COPY run_cpp.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/run_cpp.sh
ENTRYPOINT ["/usr/local/bin/run_cpp.sh"]