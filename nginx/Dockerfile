#FROM python:3.5-slim
FROM nginx:alpine

#Nginx config
RUN rm -rf /etc/nginx/conf.d
COPY nginx_conf /etc/nginx/conf.d


CMD ["nginx", "-g", "daemon off;"]

