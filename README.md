### Para ejecutar el backend:
```
docker build -t . rag-colombia
docker run -p 8000:8000 rag-colombia
```


### Para ejecutar el frontend:

```npx http-server ./frontend -p 3000```

El frontend asume que el backend esta corriendo en `localhost:8000`