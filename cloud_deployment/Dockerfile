FROM nvcr.io/nvidia/pytorch:24.05-py3  

RUN apt-get update && apt-get install -y curl git && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone Pokemon Showdown and install
RUN git clone https://github.com/smogon/pokemon-showdown.git && \
    cd pokemon-showdown && npm install && \
    cp config/config-example.js config/config.js

# Clone RL repos
RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git && \
    cd cares_reinforcement_learning && pip install -r requirements.txt && pip install -e .

RUN git clone https://github.com/UoA-CARES/showdown_gym.git && \
    cd showdown_gym && pip install -r requirements.txt && pip install -e .

RUN git clone https://github.com/UoA-CARES/gymnasium_envrionments.git && \
    cd gymnasium_envrionments && pip install -r requirements.txt

# Copy then normalize & chmod
COPY entrypoint.sh /entrypoint.sh

# Ensure LF (no CRLF) without adding a new dependency
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
