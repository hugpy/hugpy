# hugpy-chat HF Space — thin nginx reverse-proxy mirror of dev.hugpy.ai/media.
# No app bundle is shipped: the UI + API are proxied live from dev (see nginx.conf).
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 7860
CMD ["nginx", "-g", "daemon off;"]
