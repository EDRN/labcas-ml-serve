#!/bin/sh
#
# Non-containerized entrypoint for Nginx
#
# We need Nginx because we aren't allowed to reverse-proxy to http:, only https:, and
# including certificate support in Ray Serve is an unknown.

PATH=/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin:${PATH}
export PATH

: ${ML_SERVE_HOME:?✋ The environment variable ML_SERVE_HOME is required}
: ${NGINX_ETC:?✋ The environment variable NGINX_ETC is required}

if [ \! -f "${ML_SERVE_HOME}/etc/nginx.conf.in" ]; then
    echo "‼️ nginx.conf.in is not found; is ML_SERVE_HOME set correctly?" 1>&2
    exit -2
fi

CERT_CN=${CERT_CN:-localhost}
CERT_DAYS=${CERT_DAYS:-365}

echo "💁‍♀️ CERT_CN is ${CERT_CN}" 1>&2

rm -f ${ML_SERVE_HOME}/self.key ${ML_SERVE_HOME}/self.cert
openssl req -nodes -x509 -days ${CERT_DAYS} -newkey rsa:2048 -keyout ${ML_SERVE_HOME}/self.key \
    -out ${ML_SERVE_HOME}/self.cert -subj "/C=US/ST=California/L=Pasadena/O=Caltech/CN=${CERT_CN}"
rm -f ${ML_SERVE_HOME}/nginx.conf
install -d ${ML_SERVE_HOME}/var ${ML_SERVE_HOME}/var/log ${ML_SERVE_HOME}/var/log/nginx ${ML_SERVE_HOME}/var/run

for d in bodies proxy fastcgi uwsgi scgi; do
    install -d ${ML_SERVE_HOME}/var/$d
done

envsubst '$ML_SERVE_HOME $NGINX_ETC' < ${ML_SERVE_HOME}/etc/nginx.conf.in > ${ML_SERVE_HOME}/nginx.conf
exec nginx -g "daemon off;" -c ${ML_SERVE_HOME}/nginx.conf
