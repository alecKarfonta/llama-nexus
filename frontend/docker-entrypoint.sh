#!/bin/sh

echo "🚀 Starting LlamaCPP Frontend..."
echo "📊 React application with Material-UI"
echo "🔌 API proxy configured for llamacpp-api:8080"

# Replace environment variables in built files if needed
# This allows runtime configuration without rebuilding
if [ -n "$VITE_API_BASE_URL" ]; then
    echo "🔧 Configuring API base URL: $VITE_API_BASE_URL"
    find /usr/share/nginx/html -name "*.js" -exec sed -i "s|http://localhost:8600|$VITE_API_BASE_URL|g" {} \;
fi

if [ -n "$VITE_BACKEND_URL" ]; then
    echo "🔧 Configuring Backend URL: $VITE_BACKEND_URL"
    find /usr/share/nginx/html -name "*.js" -exec sed -i "s|http://localhost:8700|$VITE_BACKEND_URL|g" {} \;
fi

# Start nginx
echo "✅ Starting nginx server on port 80"
exec "$@"
