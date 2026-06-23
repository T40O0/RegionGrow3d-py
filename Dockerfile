# RegionGrow3D Python port — runtime container
#
# Builds a self-contained image that runs the Streamlit Web UI by default,
# or the CLI driver if invoked with arguments.
#
# Build:    docker build -t region3d:latest .
# Run UI:   docker run --rm -p 8501:8501 -v "$(pwd)/lib:/app/lib" \
#                        -v "$(pwd)/python/output:/app/python/output" \
#                        region3d:latest
# Run CLI:  docker run --rm -v "$(pwd)/lib:/app/lib" \
#                        -v "$(pwd)/python/output:/app/python/output" \
#                        region3d:latest \
#                        python python/driver.py --soil_strength_mode 2 \
#                                                --phi_uniform 30 --coh_uniform 5

# --------- 1) Build stage: install Python deps -------------------------------
FROM python:3.13-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries needed at runtime by rasterio (wheels bundle GDAL itself but
# require some shared C runtime libs) and by matplotlib/scipy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libexpat1 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .
RUN pip install --prefix=/install/prefix -r requirements.txt


# --------- 2) Runtime stage: copy installed packages + source ---------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/python \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libexpat1 \
       libgomp1 \
       # --- GRASS GIS + build tools so g.extension can install r.slopeunits ---
       grass-core \
       grass-dev \
       build-essential \
       git \
    && rm -rf /var/lib/apt/lists/*

# Install the r.slopeunits addon (reference slope-unit algorithm, Alvioli et
# al.) into the image. On Linux g.extension builds it from the GRASS addons
# source (the Windows precompiled-addon route 404s for some GRASS builds).
# A throwaway temporary project hosts the g.extension call.
RUN grass --tmp-location EPSG:4326 --exec \
        g.extension extension=r.slopeunits operation=add

# Patch an upstream bug in r.slopeunits.optimize: its cleanup() calls
# `grass.utils.try_rmdir`, but the module only imports `grass.script as gs`
# (NameError) and `grass.utils.try_rmdir` no longer exists in GRASS 8.4+.
# Replace it with a stdlib shutil removal so the optimizer can run.
RUN sed -i 's|grass\.utils\.try_rmdir(path)|__import__("shutil").rmtree(path, ignore_errors=True)|' \
        /root/.grass8/addons/scripts/r.slopeunits.optimize

# Copy the prefix-installed Python packages from the build stage
COPY --from=build /install/prefix /usr/local

WORKDIR /app

# Copy source. `.dockerignore` keeps the image small.
COPY . .

# The lib/ and python/output directories should be mounted at run-time so the
# user's DEM and result files persist outside the container.
VOLUME ["/app/lib", "/app/python/output", "/app/python/output_webui"]

EXPOSE 8501

# Default command: launch the Streamlit UI on 0.0.0.0:8501 (override with
# `docker run ... python python/driver.py ...` to use the CLI directly).
CMD ["streamlit", "run", "python/gui.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
