FROM node:18-alpine AS build

WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# [2026-10-15] | INFO | FRONTEND.DOCKERFILE | Template nginx.conf so the
# llamacpp API key is injected at container start (official nginx envsubst
# mechanism: /etc/nginx/templates/*.conf.template -> /etc/nginx/conf.d/).
COPY nginx.conf /etc/nginx/templates/default.conf.template
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
