FROM node:18-alpine

WORKDIR /app

# Copy package manifest, then copy the source and install dependencies.
# Some repos don't include package-lock.json; copy only package.json to avoid failures.
COPY frontend/package.json ./frontend/package.json
COPY frontend /app/frontend
WORKDIR /app/frontend
RUN apk add --no-cache python3 make g++ \
 && npm install --legacy-peer-deps

# For dev convenience we expose Vite dev server; for production build adjust accordingly
EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
