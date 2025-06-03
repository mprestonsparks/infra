# Infrastructure Development Container
FROM docker:dind

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment variables
ENV DOCKER_BUILDKIT=1
ENV COMPOSE_DOCKER_CLI_BUILD=1
ENV PATH="${PATH}:/home/vscode/.local/bin"

# Install system dependencies
RUN apk update \
    && apk add --no-cache \
        # Essential development tools
        bash \
        curl \
        wget \
        git \
        make \
        jq \
        tree \
        htop \
        vim \
        nano \
        # Network tools
        net-tools \
        iputils \
        # Additional utilities
        openssh-client \
        sudo \
        ca-certificates \
        gnupg \
        # Python for scripting
        python3 \
        python3-dev \
        py3-pip \
        # Build tools
        build-base \
        linux-headers \
        # Vault CLI dependencies
        unzip

# Install Docker Compose (standalone)
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install Terraform
RUN wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg \
    && TERRAFORM_VERSION=$(curl -s https://api.github.com/repos/hashicorp/terraform/releases/latest | jq -r .tag_name | sed 's/v//') \
    && wget https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip \
    && unzip terraform_${TERRAFORM_VERSION}_linux_amd64.zip \
    && mv terraform /usr/local/bin/ \
    && rm terraform_${TERRAFORM_VERSION}_linux_amd64.zip

# Install Vault CLI
RUN VAULT_VERSION=$(curl -s https://api.github.com/repos/hashicorp/vault/releases/latest | jq -r .tag_name | sed 's/v//') \
    && wget https://releases.hashicorp.com/vault/${VAULT_VERSION}/vault_${VAULT_VERSION}_linux_amd64.zip \
    && unzip vault_${VAULT_VERSION}_linux_amd64.zip \
    && mv vault /usr/local/bin/ \
    && rm vault_${VAULT_VERSION}_linux_amd64.zip

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install Helm
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install additional tools
RUN apk add --no-cache \
        # YAML processing
        yq \
        # Text processing
        grep \
        sed \
        awk \
        # Archive tools
        tar \
        gzip

# Create vscode user
RUN addgroup -g 1000 vscode \
    && adduser -u 1000 -G vscode -s /bin/bash -D vscode

# Create directories
RUN mkdir -p /workspaces/infra \
    && mkdir -p /data \
    && mkdir -p /logs \
    && mkdir -p /home/vscode/.local/bin

# Set up user permissions
RUN echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Switch to vscode user
USER vscode

# Install Python tools
RUN python3 -m pip install --user --upgrade pip \
    && python3 -m pip install --user \
        # Infrastructure as Code tools
        ansible \
        # Cloud SDKs
        boto3 \
        azure-cli \
        google-cloud-storage \
        # Utilities
        pyyaml \
        jinja2 \
        requests \
        # Development tools
        black \
        ruff \
        pytest

# Create useful aliases and shell configuration
RUN echo 'alias ll="ls -alF"' >> ~/.bashrc \
    && echo 'alias la="ls -A"' >> ~/.bashrc \
    && echo 'alias l="ls -CF"' >> ~/.bashrc \
    && echo 'alias ..="cd .."' >> ~/.bashrc \
    && echo 'alias dc="docker-compose"' >> ~/.bashrc \
    && echo 'alias dcu="docker-compose up"' >> ~/.bashrc \
    && echo 'alias dcd="docker-compose down"' >> ~/.bashrc \
    && echo 'alias dcb="docker-compose build"' >> ~/.bashrc \
    && echo 'alias dps="docker ps"' >> ~/.bashrc \
    && echo 'alias di="docker images"' >> ~/.bashrc \
    && echo 'alias tf="terraform"' >> ~/.bashrc \
    && echo 'alias tfi="terraform init"' >> ~/.bashrc \
    && echo 'alias tfp="terraform plan"' >> ~/.bashrc \
    && echo 'alias tfa="terraform apply"' >> ~/.bashrc \
    && echo 'alias tfd="terraform destroy"' >> ~/.bashrc \
    && echo 'alias k="kubectl"' >> ~/.bashrc \
    && echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc \
    && echo 'export VAULT_ADDR=http://host.docker.internal:8200' >> ~/.bashrc

# Set up git configuration
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global core.autocrlf input \
    && git config --global core.eol lf

# Create workspace directory and set permissions
RUN sudo mkdir -p /workspaces/infra \
    && sudo chown -R vscode:vscode /workspaces/infra

# Set the working directory
WORKDIR /workspaces/infra

# Default command
CMD ["sleep", "infinity"]