# Phase 4 UI in a container: build with Node, serve the static files with nginx.
# Build context: phase4-observability/ui/
#   docker build -f phase8-docker/ui/Dockerfile -t lab/ui phase4-observability/ui
FROM node:24-alpine AS build
WORKDIR /ui
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Baked in at build time. The browser runs on the host, so this is the API's
# published host port, not a container name.
ARG VITE_API_BASE=http://127.0.0.1:8010
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.29-alpine
COPY --from=build /ui/dist /usr/share/nginx/html
EXPOSE 80
