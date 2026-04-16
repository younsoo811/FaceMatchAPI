#See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base

USER root

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    libjpeg62 \
    libpng16-16 \
    libharfbuzz0b \
    libgtk-3-0 \
    libdrm2 \
    libatomic1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

WORKDIR /src
COPY . .
RUN dotnet restore

RUN dotnet publish \
    -c Release \
    -o /app/publish \
    -r linux-x64 \
    --self-contained false \
    /p:CopyLocalLockFileAssemblies=true

FROM base AS final
WORKDIR /app
COPY --from=build /app/publish .

RUN ln -s /app/libOpenCvSharpExtern.so /app/OpenCvSharpExtern.so

ENTRYPOINT ["dotnet", "FaceMatchAPI.dll"]