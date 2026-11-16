// ==UserScript==
// @name         FaceMatch
// @namespace    facematch.local.tool
// @updateURL    https://raw.githubusercontent.com/Gusyevaa/facematch/main/facematch.user.js
// @downloadURL  https://raw.githubusercontent.com/Gusyevaa/facematch/main/facematch.user.js
// @version      1.2.0
// @description  Compara localmente (100% no navegador) a foto do cliente com as fotos da Lista Negativa e exibe um indicador visual de similaridade facial. Não toma decisões — apenas auxilia a análise humana.
// @author       você
// @match        *://*.acsoluti.com.br/certdig/getfile/idarquivo/*
// @grant        GM_xmlhttpRequest
// @grant        GM.xmlHttpRequest
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM.setValue
// @grant        GM.getValue
// @connect      safimagens.acsoluti.com.br
// @connect      acsoluti.com.br
// @run-at       document-idle
// @require      https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js
// ==/UserScript==

/*
 * ============================================================================
 * FaceMatch
 * ----------------------------------------------------------------------------
 * IMPORTANTE (leia antes de usar):
 *
 * 1) O @match acima cobre todo o domínio *.acsoluti.com.br, já que a URL
 *    específica da tela de Lista Negativa não segue um padrão fixo. Para
 *    compensar isso, o script só ATIVA de fato (PageGuard, mais abaixo)
 *    quando detecta, na própria página, que está na tela certa — pelo
 *    título da página e/ou pela presença do elemento da foto do cliente.
 *    Em qualquer outra página do domínio, o script fica ocioso e não
 *    carrega os modelos nem faz nada.
 *
 * 2) Este script carrega os PESOS DO MODELO de reconhecimento facial (arquivos
 *    estáticos, sem nenhuma imagem sua) a partir de um CDN público (jsDelivr),
 *    via @require abaixo. Nenhuma foto, embedding ou dado do cliente é
 *    enviado para fora do navegador em nenhum momento — todo o cálculo
 *    (detecção, alinhamento, embedding, similaridade) ocorre localmente,
 *    em memória, usando TensorFlow.js.
 *
 * 3) Sobre backend (WebGL/WASM/WebGPU): em testes reais, o backend WASM
 *    do TensorFlow.js causou erros durante a detecção facial ("Cannot read
 *    properties of undefined (reading 'type')"), pois não suporta bem
 *    algumas operações usadas pelos modelos. Por isso, por padrão o script
 *    usa apenas WebGL (aceleração por GPU, bem suportado pelo face-api.js)
 *    com fallback automático para CPU — ambos garantidos de funcionar.
 *    Se quiser tentar WASM/WebGPU mesmo assim, mude
 *    CONFIG.TRY_EXPERIMENTAL_BACKENDS para true e adicione de volta o
 *    @require do tfjs-backend-wasm no cabeçalho — mas espere possíveis
 *    falhas, e o script já tem verificação de sanidade para pular
 *    automaticamente para WebGL/CPU se isso acontecer.
 * ============================================================================
 */

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // CONFIGURAÇÃO
    // ------------------------------------------------------------------
    const CONFIG = {
        // Seletores fixos, conforme a estrutura conhecida da página
        CLIENT_SELECTOR: "#userinfo-picture img",
        NEGATIVE_SELECTOR: ".picture-box img",

        AUTO_START: true,

        // Se true, o badge mostra o rótulo de texto (ALTÍSSIMA, ALTA, ...).
        // Se false, mostra a porcentagem numérica (ex.: "72%").
        // A porcentagem exata continua disponível no tooltip (mouse em cima)
        // nos dois casos.
        USE_TEXT_LABELS: true,

        // Processamento em lotes para não travar a UI. Valores maiores
        // aceleram bastante, já que boa parte do tempo é rede (baixar a
        // imagem) e não CPU/GPU — então dá pra ter várias em andamento.
        BATCH_SIZE: 8,
        BATCH_DELAY_MS: 5,

        // Modelos do face-api (vladmandic fork), servidos como arquivos
        // estáticos via CDN — apenas pesos, nenhum dado enviado.
        MODEL_URL: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/",

        // Detector: o 'ssdMobilenetv1' é mais preciso, mas em testes reais
        // ficou lento/travado demais em algumas máquinas (modelo maior pra
        // baixar + detecção bem mais pesada, principalmente sem GPU via
        // WebGL). Por segurança e confiabilidade, usamos
        // 'tinyFaceDetector' — mas com resolução interna maior (384 em vez
        // de 224) para recuperar parte da precisão sem o risco de travar.
        DETECTOR: "tinyFaceDetector",

        // Calibração da similaridade de cosseno → porcentagem.
        //
        // Por que isso importa: embeddings faciais (128 dimensões) desse
        // modelo raramente têm cosseno abaixo de ~0.3-0.4 mesmo entre
        // pessoas diferentes — a faixa útil de verdade fica concentrada
        // perto de 1.0. Usar a escala completa de -1 a 1 (ou até 0 a 1)
        // faz TUDO cair perto de 90-100%, mesmo rostos bem diferentes.
        //
        // Valores calibrados com teste real (fotos pravatar.cc): pessoas
        // diferentes deram cosseno entre 0.80 e 0.88; a mesma pessoa deu
        // 0.9976 — uma separação bem clara. MIN/MAX abaixo refletem essa
        // faixa observada, com uma margem de segurança. Se depois de usar
        // no site real você notar que a maioria das fotos de pessoas
        // diferentes ainda fica muito alta (ou muito baixa), ajuste esses
        // dois números com base nos valores de "cosine=0.xx" que aparecem
        // no console (DEBUG) para os seus casos reais.
        SIMILARITY_CALIBRATION: {
            MIN: 0.8563, // cosseno neste valor ou abaixo → 0%
            MAX: 0.9778, // cosseno neste valor ou acima → 100%
        },

        // Backends WASM/WebGPU são experimentais e causaram erros
        // ("Cannot read properties of undefined (reading 'type')") em testes
        // reais — o tfjs-backend-wasm não suporta bem todas as operações
        // usadas pelos modelos de detecção facial. Por isso ficam desligados
        // por padrão. WebGL é o backend garantido de funcionar (GPU via
        // WebGL), com CPU como fallback automático.
        TRY_EXPERIMENTAL_BACKENDS: false,

        // Faixas de cor + rótulo de texto mostrado no badge.
        // Para mudar os textos, é só editar o "label" aqui.
        THRESHOLDS: [
            { min: 90, max: 100, color: "#e53935", label: "ALTÍSSIMA" }, // vermelho
            { min: 70, max: 89.999, color: "#fb8c00", label: "ALTA" }, // laranja
            { min: 50, max: 69.999, color: "#fdd835", label: "MÉDIA" }, // amarelo
            { min: 30, max: 49.999, color: "#43a047", label: "BAIXA" }, // verde
            { min: 0, max: 29.999, color: "#9e9e9e", label: "BAIXÍSSIMA" }, // cinza
        ],

        // Rótulos dos casos especiais (não são faixas de similaridade).
        LABEL_NO_FACE: "N/A",
        LABEL_ERROR: "Erro",

        // Borda colorida ao redor da própria foto (além do badge).
        // pct >= BORDER_THRESHOLD_PCT → verde | pct < BORDER_THRESHOLD_PCT → vermelha
        BORDER_THRESHOLD_PCT: 50,
        BORDER_COLOR_MATCH: "#00e676", // verde vibrante
        BORDER_COLOR_NOMATCH: "#ff1744", // vermelho vibrante
        BORDER_WIDTH_PX: 4,

        DEBUG: true,

        // Ferramenta de calibração: depois de processar a lista, compara
        // todas as fotos da lista ENTRE SI (não só com o cliente) e mostra
        // no console os pares mais parecidos. Como o mesmo fraudador pode
        // aparecer várias vezes na lista (ângulos/dias diferentes), os
        // pares com cosseno mais alto são bons candidatos a "mesma pessoa,
        // foto diferente" — dados reais do seu ambiente pra calibrar
        // SIMILARITY_CALIBRATION, sem precisar esperar um caso de cliente
        // batendo com fraudador (que é raro por natureza).
        // Deixe true só enquanto estiver calibrando — é uma comparação a
        // mais pra cada par de fotos (custo cresce rápido com o tamanho
        // da lista), então desligue depois de coletar os dados.
        DEBUG_PAIRWISE_ANALYSIS: false,

        // Guarda de página: o script só ativa se pelo menos UMA das
        // condições abaixo for satisfeita.
        PAGE_GUARD: {
            // Trecho(s) que devem aparecer no <title> da página (case-insensitive).
            TITLE_KEYWORDS: ["Lista Negativa"],
            // Seletor cuja presença já basta para considerar "é a tela certa"
            // (normalmente o mesmo da foto do cliente).
            REQUIRE_SELECTOR: "#userinfo-picture img",
            // Tempo máximo esperando a página carregar (útil se o título ou
            // o elemento aparecem via JS/AJAX depois do carregamento inicial).
            MAX_WAIT_MS: 8000,
            POLL_INTERVAL_MS: 300,
        },
    };

    // ------------------------------------------------------------------
    // LOGGER
    // ------------------------------------------------------------------
    const Logger = {
        log(...args) {
            if (CONFIG.DEBUG) console.log("[FaceMatch]", ...args);
        },
        warn(...args) {
            if (CONFIG.DEBUG) console.warn("[FaceMatch]", ...args);
        },
        error(...args) {
            console.error("[FaceMatch]", ...args);
        },
    };

    // ------------------------------------------------------------------
    // STATUS PANEL — agora só loga no console (CONFIG.DEBUG), sem desenhar
    // nada na página. Mantido como no-op para não precisar mexer nas
    // chamadas StatusPanel.update(...) espalhadas pelo código.
    // ------------------------------------------------------------------
    const StatusPanel = {
        update(msg, isError) {
            if (isError) Logger.error(msg);
            else Logger.log(msg);
        },
    };

    // ------------------------------------------------------------------
    // IMAGE FETCHER — busca imagens de outra origem via GM_xmlhttpRequest,
    // contornando a restrição de CORS que "suja" o canvas e bloqueia a
    // leitura de pixels necessária para o reconhecimento facial.
    // ------------------------------------------------------------------
    const ImageFetcher = {
        _gmRequest(config) {
            if (typeof GM !== "undefined" && GM.xmlHttpRequest) {
                return GM.xmlHttpRequest(config);
            }
            if (typeof GM_xmlhttpRequest !== "undefined") {
                return GM_xmlhttpRequest(config);
            }
            throw new Error(
                "GM_xmlhttpRequest indisponível — verifique se @grant GM_xmlhttpRequest está no cabeçalho do script."
            );
        },

        fetchAsBlob(url) {
            return new Promise((resolve, reject) => {
                try {
                    this._gmRequest({
                        method: "GET",
                        url,
                        responseType: "blob",
                        onload: (response) => {
                            if (response.status >= 200 && response.status < 300) {
                                resolve(response.response);
                            } else {
                                reject(new Error(`HTTP ${response.status} ao buscar imagem: ${url}`));
                            }
                        },
                        onerror: () => reject(new Error(`Erro de rede ao buscar imagem: ${url}`)),
                        ontimeout: () => reject(new Error(`Timeout ao buscar imagem: ${url}`)),
                    });
                } catch (err) {
                    reject(err);
                }
            });
        },

        blobToDataUrl(blob) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error("Falha ao converter imagem para Data URL"));
                reader.readAsDataURL(blob);
            });
        },

        /**
         * Baixa a imagem da URL informada (contornando CORS) e retorna um
         * elemento <img> offscreen, já carregado, pronto para ser usado na
         * detecção facial local. A imagem visível na página não é alterada.
         */
        async loadOffscreenCopy(url) {
            const blob = await this.fetchAsBlob(url);
            const dataUrl = await this.blobToDataUrl(blob);

            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error(`Falha ao carregar cópia local da imagem: ${url}`));
                img.src = dataUrl;
            });
        },
    };

    // ------------------------------------------------------------------
    // KEEP ALIVE — mantém um áudio inaudível tocando para que o navegador
    // não aplique o throttling agressivo de timers que ele reserva para
    // abas em segundo plano "inativas" (Chrome, por exemplo, passa a
    // rodar setTimeout só 1x por minuto depois de alguns minutos de aba
    // oculta — abas que reproduzem áudio ficam isentas dessa regra).
    // Não precisa de permissão nenhuma; é só Web Audio API padrão.
    // ------------------------------------------------------------------
    const KeepAlive = {
        _ctx: null,

        init() {
            const start = () => {
                if (this._ctx) return;
                try {
                    const AudioCtx = window.AudioContext || window.webkitAudioContext;
                    this._ctx = new AudioCtx();

                    const gain = this._ctx.createGain();
                    gain.gain.value = 0; // completamente silencioso

                    const oscillator = this._ctx.createOscillator();
                    oscillator.frequency.value = 20000;
                    oscillator.connect(gain);
                    gain.connect(this._ctx.destination);
                    oscillator.start();

                    Logger.log("Áudio silencioso ativo (evita throttling de aba em segundo plano).");
                } catch (err) {
                    Logger.warn("Não foi possível iniciar o keep-alive de áudio:", err);
                }
                document.removeEventListener("click", start);
                document.removeEventListener("keydown", start);
            };

            // Navegadores exigem uma interação do usuário antes de permitir
            // áudio (política de autoplay). Como você já está navegando na
            // página, o primeiro clique ou tecla ativa isso automaticamente.
            document.addEventListener("click", start, { once: true });
            document.addEventListener("keydown", start, { once: true });
        },
    };

    // ------------------------------------------------------------------
    // PREPROCESSOR — melhora a imagem ANTES da detecção facial.
    //
    // Importante ser honesto sobre o que isso faz de verdade:
    // - Aumenta imagens pequenas para um tamanho mínimo, o que ajuda o
    //   detector a ENCONTRAR e ALINHAR o rosto com mais precisão (rostos
    //   muito pequenos em pixels são mais difíceis de localizar bem).
    // - Aplica um leve ajuste de contraste/brilho, que ajuda o detector a
    //   distinguir bordas do rosto em fotos escuras/desbotadas.
    // - NÃO inventa detalhe que não existe na foto original. Se a foto é
    //   muito baixa resolução ou o rosto está borrado/de perfil, isso
    //   melhora a chance de detecção correta, mas não resolve o limite
    //   físico de informação disponível na imagem.
    // ------------------------------------------------------------------
    const Preprocessor = {
        MIN_DIMENSION: 320, // tamanho mínimo (em pixels) do lado menor da imagem
        MAX_DIMENSION: 640, // tamanho máximo do lado MAIOR — evita canvas gigante e lento
        ENABLE_CONTRAST_ENHANCEMENT: true, // desligue aqui se ainda estiver lento

        /**
         * Recebe uma imagem (HTMLImageElement) e devolve um <canvas> pronto
         * para ser usado na detecção facial: redimensionado (pequeno demais
         * vira maior, grande demais vira menor) e com contraste/brilho
         * levemente realçados. Limitar o tamanho MÁXIMO é importante:
         * fotos de captura de documento costumam vir em alta resolução, e
         * processar/filtrar um canvas gigante pode travar o navegador por
         * bastante tempo antes mesmo da detecção facial começar.
         */
        process(img) {
            const t0 = CONFIG.DEBUG ? performance.now() : 0;

            const srcW = img.naturalWidth || img.width;
            const srcH = img.naturalHeight || img.height;
            const minSide = Math.min(srcW, srcH);
            const maxSide = Math.max(srcW, srcH);

            let scale = 1;
            if (minSide < this.MIN_DIMENSION) {
                scale = this.MIN_DIMENSION / minSide;
            }
            // O limite de tamanho máximo tem prioridade: nunca deixa o
            // canvas passar de MAX_DIMENSION no lado maior, mesmo que isso
            // signifique cancelar parte do upscale acima.
            if (maxSide * scale > this.MAX_DIMENSION) {
                scale = this.MAX_DIMENSION / maxSide;
            }

            const w = Math.round(srcW * scale);
            const h = Math.round(srcH * scale);

            const canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;

            const ctx = canvas.getContext("2d");
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            if (this.ENABLE_CONTRAST_ENHANCEMENT) {
                // Leve realce de contraste/brilho — ajuda em fotos escuras
                // ou desbotadas. OBS: ctx.filter pode ser lento em canvas
                // grandes; por isso limitamos o tamanho ANTES de aplicar.
                ctx.filter = "contrast(1.15) brightness(1.05)";
            }
            ctx.drawImage(img, 0, 0, w, h);

            if (CONFIG.DEBUG) {
                Logger.log(
                    `Preprocessor: ${srcW}x${srcH} -> ${w}x${h} em ${(performance.now() - t0).toFixed(0)}ms`
                );
            }

            return canvas;
        },
    };

    // ------------------------------------------------------------------
    // PERSISTENT CACHE — guarda a assinatura facial (embedding) de cada
    // foto de fraudador PERMANENTEMENTE, usando o armazenamento do próprio
    // Tampermonkey (GM_setValue/GM_getValue). Isso sobrevive entre
    // consultas diferentes, dias diferentes, etc. — como as fotos da
    // Lista Negativa se repetem bastante entre consultas, a maioria delas
    // só precisa ser baixada e processada UMA VEZ na vida; depois disso o
    // script já "sabe" a resposta na hora. Tudo continua 100% local: nada
    // sai do seu navegador, é só o armazenamento próprio da extensão.
    // ------------------------------------------------------------------
    const PersistentCache = {
        _hasAsyncGM() {
            return typeof GM !== "undefined" && GM.getValue && GM.setValue;
        },

        async get(key) {
            try {
                if (this._hasAsyncGM()) {
                    const raw = await GM.getValue(key, null);
                    return raw ? JSON.parse(raw) : null;
                }
                if (typeof GM_getValue !== "undefined") {
                    const raw = GM_getValue(key, null);
                    return raw ? JSON.parse(raw) : null;
                }
            } catch (err) {
                Logger.warn(`Falha ao ler cache persistente (${key}):`, err);
            }
            return null;
        },

        async set(key, value) {
            try {
                const raw = JSON.stringify(value);
                if (this._hasAsyncGM()) {
                    await GM.setValue(key, raw);
                    return;
                }
                if (typeof GM_setValue !== "undefined") {
                    GM_setValue(key, raw);
                }
            } catch (err) {
                Logger.warn(`Falha ao gravar cache persistente (${key}):`, err);
            }
        },
    };

    // ------------------------------------------------------------------
    // PAGE GUARD — só deixa o script ativar na tela certa
    // ------------------------------------------------------------------
    const PageGuard = {
        matchesNow() {
            const { TITLE_KEYWORDS, REQUIRE_SELECTOR } = CONFIG.PAGE_GUARD;

            const titleMatches = TITLE_KEYWORDS.some((kw) =>
                document.title.toLowerCase().includes(kw.toLowerCase())
            );
            const selectorMatches =
                !!REQUIRE_SELECTOR && !!document.querySelector(REQUIRE_SELECTOR);

            return titleMatches || selectorMatches;
        },

        /**
         * Fica verificando periodicamente (para telas que carregam título
         * ou conteúdo via AJAX/JS depois do carregamento inicial da página)
         * até encontrar um match ou estourar o tempo limite.
         */
        async waitForMatch() {
            const { MAX_WAIT_MS, POLL_INTERVAL_MS } = CONFIG.PAGE_GUARD;
            const start = Date.now();

            while (Date.now() - start < MAX_WAIT_MS) {
                if (this.matchesNow()) return true;
                await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
            }

            return this.matchesNow(); // última checagem
        },
    };

    // ------------------------------------------------------------------
    // BACKEND (TensorFlow.js) — WebGL/CPU por padrão (garantido); WASM/WebGPU
    // apenas se CONFIG.TRY_EXPERIMENTAL_BACKENDS = true (ver aviso no CONFIG)
    // ------------------------------------------------------------------
    const BackendManager = {
        async init() {
            const tf = faceapi.tf;
            const candidates = CONFIG.TRY_EXPERIMENTAL_BACKENDS
                ? ["webgpu", "wasm", "webgl", "cpu"]
                : ["webgl", "cpu"];

            if (CONFIG.TRY_EXPERIMENTAL_BACKENDS) {
                try {
                    if (tf.wasm && tf.wasm.setWasmPaths) {
                        tf.wasm.setWasmPaths(
                            "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/"
                        );
                    }
                } catch (err) {
                    Logger.warn("Não foi possível configurar caminho do WASM:", err);
                }
            }

            for (const backendName of candidates) {
                try {
                    if (!tf.engine().registryFactory[backendName] &&
                        !tf.findBackend(backendName)) {
                        continue;
                    }
                    await tf.setBackend(backendName);
                    await tf.ready();

                    // Teste real: roda uma operação mínima para garantir que o
                    // backend realmente funciona com os kernels necessários,
                    // não só que foi "setado". Isso evita cair num backend
                    // que aceita ser ativado mas falha durante a detecção.
                    await this._sanityCheck(tf);

                    Logger.log(`Backend ativo: ${tf.getBackend()}`);
                    return tf.getBackend();
                } catch (err) {
                    Logger.warn(`Falha ao ativar/validar backend "${backendName}", tentando próximo...`, err);
                }
            }

            // Último recurso: deixa o face-api escolher automaticamente.
            await tf.ready();
            Logger.log(`Backend final (automático): ${tf.getBackend()}`);
            return tf.getBackend();
        },

        /**
         * Executa uma operação simples de convolução (mesma família de
         * operação usada pelos modelos de detecção facial) para confirmar
         * que o backend ativo suporta os kernels necessários.
         */
        async _sanityCheck(tf) {
            const input = tf.zeros([1, 8, 8, 3]);
            const filter = tf.zeros([3, 3, 3, 1]);
            const result = tf.conv2d(input, filter, 1, "same");
            await result.data();
            input.dispose();
            filter.dispose();
            result.dispose();
        },
    };

    // ------------------------------------------------------------------
    // FACE DETECTOR — carrega modelos e executa detecção + landmarks
    // ------------------------------------------------------------------
    const FaceDetector = {
        _ready: false,

        async loadModels() {
            if (this._ready) return;

            const url = CONFIG.MODEL_URL;
            const loadPromise = Promise.all([
                CONFIG.DETECTOR === "ssdMobilenetv1"
                    ? faceapi.nets.ssdMobilenetv1.loadFromUri(url)
                    : faceapi.nets.tinyFaceDetector.loadFromUri(url),
                faceapi.nets.faceLandmark68Net.loadFromUri(url),
                faceapi.nets.faceRecognitionNet.loadFromUri(url),
            ]);

            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(
                    () => reject(new Error("Timeout de 30s ao carregar modelos (CDN lento ou bloqueado)")),
                    30000
                )
            );

            await Promise.race([loadPromise, timeoutPromise]);

            this._ready = true;
            Logger.log("Modelos de reconhecimento facial carregados.");
        },

        _getDetectorOptions() {
            return CONFIG.DETECTOR === "ssdMobilenetv1"
                ? new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 })
                : new faceapi.TinyFaceDetectorOptions({ inputSize: 384, scoreThreshold: 0.4 });
        },

        /**
         * Detecta um único rosto na imagem e retorna a detecção completa
         * (bounding box + landmarks + descritor facial de 128 dimensões).
         * Retorna null se nenhum rosto for encontrado.
         */
        async detectFace(source) {
            await this.loadModels();

            // O elemento pode ser uma <img> (raramente, se não passar pelo
            // Preprocessor) ou um <canvas> (o caso normal, vindo do
            // Preprocessor.process()). Canvas nunca dispara 'load'/'error'
            // — está sempre pronto assim que é desenhado — então só espera
            // pelo carregamento quando for de fato uma <img> ainda não
            // carregada. Sem esse cuidado, a espera por 'load' num canvas
            // nunca resolve e trava para sempre (foi exatamente esse o bug).
            const isImgElement = typeof HTMLImageElement !== "undefined" && source instanceof HTMLImageElement;

            if (isImgElement && (!source.complete || source.naturalWidth === 0)) {
                await new Promise((resolve, reject) => {
                    source.addEventListener("load", resolve, { once: true });
                    source.addEventListener("error", reject, { once: true });
                });
            }

            const result = await faceapi
                .detectSingleFace(source, this._getDetectorOptions())
                .withFaceLandmarks()
                .withFaceDescriptor();

            return result || null;
        },
    };

    // ------------------------------------------------------------------
    // EMBEDDING GENERATOR — extrai a assinatura facial (embedding)
    // ------------------------------------------------------------------
    const EmbeddingGenerator = {
        /**
         * A partir do resultado de FaceDetector.detectFace(), retorna o
         * embedding (Float32Array de 128 posições) já alinhado.
         */
        extractEmbedding(detectionResult) {
            if (!detectionResult || !detectionResult.descriptor) return null;
            return detectionResult.descriptor;
        },
    };

    // ------------------------------------------------------------------
    // SIMILARITY CALCULATOR
    // ------------------------------------------------------------------
    const SimilarityCalculator = {
        cosineSimilarity(a, b) {
            let dot = 0, normA = 0, normB = 0;
            for (let i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            if (normA === 0 || normB === 0) return 0;
            return dot / (Math.sqrt(normA) * Math.sqrt(normB));
        },

        /**
         * Converte a similaridade de cosseno para uma escala visual de
         * 0 a 100, usando os limites de CONFIG.SIMILARITY_CALIBRATION
         * (ver comentário no CONFIG explicando o porquê da recalibração).
         */
        toPercentage(cosineSim) {
            const { MIN, MAX } = CONFIG.SIMILARITY_CALIBRATION;
            const normalized = (cosineSim - MIN) / (MAX - MIN);
            const pct = Math.max(0, Math.min(1, normalized)) * 100;

            if (CONFIG.DEBUG) {
                Logger.log(`cosine=${cosineSim.toFixed(4)} -> ${pct.toFixed(1)}%`);
            }

            return Math.round(pct * 10) / 10;
        },
    };

    // ------------------------------------------------------------------
    // OVERLAY — desenha o badge de similaridade sobre cada imagem
    // ------------------------------------------------------------------
    const Overlay = {
        /**
         * Retorna a faixa (cor + rótulo) correspondente à porcentagem.
         */
        getRangeForPercent(pct) {
            return (
                CONFIG.THRESHOLDS.find((t) => pct >= t.min && pct <= t.max) ||
                CONFIG.THRESHOLDS[CONFIG.THRESHOLDS.length - 1]
            );
        },

        getColorForPercent(pct) {
            return this.getRangeForPercent(pct).color;
        },

        _ensureWrapper(imgEl) {
            if (imgEl.parentElement && imgEl.parentElement.classList.contains("facematch-wrapper")) {
                return imgEl.parentElement;
            }
            const wrapper = document.createElement("div");
            wrapper.className = "facematch-wrapper";
            wrapper.style.position = "relative";
            wrapper.style.display = "inline-block";
            imgEl.parentNode.insertBefore(wrapper, imgEl);
            wrapper.appendChild(imgEl);
            return wrapper;
        },

        _ensureBadge(wrapper) {
            let badge = wrapper.querySelector(".facematch-badge");
            if (!badge) {
                badge = document.createElement("div");
                badge.className = "facematch-badge";
                Object.assign(badge.style, {
                    position: "absolute",
                    bottom: "4px",
                    right: "4px",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    fontSize: "10px",
                    fontWeight: "bold",
                    color: "#fff",
                    fontFamily: "sans-serif",
                    textShadow: "0 1px 2px rgba(0,0,0,0.6)",
                    pointerEvents: "none",
                    zIndex: 9999,
                });
                wrapper.appendChild(badge);
            }
            return badge;
        },

        /**
         * Aplica (ou remove) a borda colorida ao redor da própria foto,
         * de acordo com o limiar definido em CONFIG.BORDER_THRESHOLD_PCT.
         * pct >= limiar → borda verde (CONFIG.BORDER_COLOR_MATCH)
         * pct <  limiar → borda vermelha (CONFIG.BORDER_COLOR_NOMATCH)
         */
        _applyBorder(imgEl, color) {
            imgEl.style.boxSizing = "border-box";
            imgEl.style.border = color ? `${CONFIG.BORDER_WIDTH_PX}px solid ${color}` : "";
        },

        showSimilarity(imgEl, pct) {
            const wrapper = this._ensureWrapper(imgEl);
            const badge = this._ensureBadge(wrapper);

            // Badge e borda usam a MESMA cor (verde/vermelho vibrante),
            // conforme o limiar — sem rótulos de texto (ALTÍSSIMA, ALTA, ...).
            const borderColor =
                pct >= CONFIG.BORDER_THRESHOLD_PCT ? CONFIG.BORDER_COLOR_MATCH : CONFIG.BORDER_COLOR_NOMATCH;

            badge.style.background = borderColor;
            badge.textContent = `${pct.toFixed(0)}%`;
            // A porcentagem exata continua acessível passando o mouse por cima.
            badge.title = `Similaridade: ${pct.toFixed(0)}% (apoio visual — não é probabilidade de fraude)`;

            this._applyBorder(imgEl, borderColor);
        },

        showNoFace(imgEl) {
            const wrapper = this._ensureWrapper(imgEl);
            const badge = this._ensureBadge(wrapper);
            badge.style.background = "#616161";
            badge.textContent = CONFIG.LABEL_NO_FACE;
            badge.title = "Nenhum rosto detectado nesta foto";
            // Sem rosto detectado: não há como avaliar match, então nenhuma
            // borda verde/vermelha é aplicada (evita falso positivo/negativo).
            this._applyBorder(imgEl, null);
        },

        showError(imgEl) {
            const wrapper = this._ensureWrapper(imgEl);
            const badge = this._ensureBadge(wrapper);
            badge.style.background = "#000";
            badge.textContent = CONFIG.LABEL_ERROR;
            badge.title = "Erro ao processar esta foto";
            this._applyBorder(imgEl, null);
        },
    };

    // ------------------------------------------------------------------
    // UTILS
    // ------------------------------------------------------------------
    const Utils = {
        /**
         * Espera até que ao menos um elemento casando com `selector` exista
         * no DOM, útil quando o conteúdo é carregado via AJAX/SPA depois
         * do carregamento inicial da página.
         */
        waitForElement(selector, timeoutMs = 15000, intervalMs = 300) {
            return new Promise((resolve) => {
                const start = Date.now();
                const check = () => {
                    const el = document.querySelector(selector);
                    if (el) return resolve(el);
                    if (Date.now() - start >= timeoutMs) return resolve(null);
                    setTimeout(check, intervalMs);
                };
                check();
            });
        },

        /**
         * Executa uma Promise com um limite de tempo — se não resolver a
         * tempo, rejeita com um erro claro em vez de travar para sempre.
         */
        withTimeout(promise, ms, message) {
            const timeout = new Promise((_, reject) =>
                setTimeout(() => reject(new Error(message || `Timeout de ${ms}ms`)), ms)
            );
            return Promise.race([promise, timeout]);
        },
    };

    // ------------------------------------------------------------------
    // MAIN — orquestra todo o fluxo
    // ------------------------------------------------------------------
    const Main = {
        clientEmbedding: null,
        processedImgs: new WeakSet(),
        offscreenCache: new Map(), // src -> HTMLImageElement, evita rebaixar duplicatas
        collectedEmbeddings: [], // { src, embedding } — só usado se DEBUG_PAIRWISE_ANALYSIS
        queue: [],
        processing: false,
        _forcedCpuFallback: false, // depois de travar uma vez, fica sempre no CPU

        /**
         * Roda a detecção facial com limite de tempo. Se travar/der timeout
         * (geralmente por causa do backend WebGL — comum quando a aba fica
         * em segundo plano, já que o navegador reduz a prioridade de
         * tarefas de GPU pra abas ocultas), troca automaticamente pro
         * backend CPU (mais lento, mas nunca trava) e tenta de novo. Uma
         * vez que isso acontece, o resto da sessão já fica direto no CPU.
         */
        async detectWithFallback(canvas, label) {
            if (this._forcedCpuFallback) {
                return FaceDetector.detectFace(canvas);
            }

            try {
                return await Utils.withTimeout(
                    FaceDetector.detectFace(canvas),
                    20000,
                    `Timeout (20s): ${label}`
                );
            } catch (err) {
                Logger.warn(
                    `Detecção travou em "${label}" com backend "${faceapi.tf.getBackend()}". Trocando para CPU e tentando de novo...`
                );
                try {
                    await faceapi.tf.setBackend("cpu");
                    await faceapi.tf.ready();
                    this._forcedCpuFallback = true;
                } catch (switchErr) {
                    Logger.error("Falha ao trocar para backend CPU:", switchErr);
                    throw err; // mantém o erro original se nem a troca funcionar
                }
                return Utils.withTimeout(
                    FaceDetector.detectFace(canvas),
                    30000,
                    `Timeout (30s) mesmo já no CPU: ${label}`
                );
            }
        },

        /**
         * Retorna uma versão da imagem segura para leitura de pixels
         * (sem "tainted canvas"). Imagens data: URI (a foto do cliente,
         * embutida na página) já são seguras e usadas diretamente. Imagens
         * de outra origem (as fotos da lista, hospedadas em
         * safimagens.acsoluti.com.br) são baixadas via GM_xmlhttpRequest
         * (contornando CORS) e carregadas como uma cópia local offscreen.
         */
        async getCorsSafeImage(imgEl) {
            const src = imgEl.currentSrc || imgEl.src;

            if (src.startsWith("data:")) {
                return imgEl;
            }

            if (this.offscreenCache.has(src)) {
                return this.offscreenCache.get(src);
            }

            const offscreenImg = await ImageFetcher.loadOffscreenCopy(src);
            this.offscreenCache.set(src, offscreenImg);
            return offscreenImg;
        },

        async init() {
            StatusPanel.update("FaceMatch iniciando...");
            KeepAlive.init();

            try {
                const backend = await BackendManager.init();
                StatusPanel.update(`Backend TensorFlow.js: ${backend}`);
            } catch (err) {
                StatusPanel.update(`Erro ao iniciar backend: ${err.message}`, true);
            }

            try {
                StatusPanel.update("Carregando modelos de reconhecimento facial...");
                await FaceDetector.loadModels();
                StatusPanel.update("Modelos carregados com sucesso.");
            } catch (err) {
                StatusPanel.update(
                    `ERRO ao carregar modelos (provável bloqueio de CDN/CSP no site): ${err.message}`,
                    true
                );
                return; // Sem modelos, não há como continuar.
            }

            await this.loadClientEmbedding();
            await this.scanAndQueueNegativeImages();
            this.observeMutations();

            StatusPanel.update("FaceMatch pronto e observando a página.");
        },

        async loadClientEmbedding() {
            StatusPanel.update(`Procurando foto do cliente ("${CONFIG.CLIENT_SELECTOR}")...`);
            const clientImg = await Utils.waitForElement(CONFIG.CLIENT_SELECTOR);

            if (!clientImg) {
                StatusPanel.update(
                    `Foto do cliente NÃO encontrada com o seletor "${CONFIG.CLIENT_SELECTOR}". Verifique se o seletor bate com o HTML real da página.`,
                    true
                );
                return;
            }
            StatusPanel.update("Foto do cliente encontrada. Processando rosto...");

            try {
                const safeImg = await this.getCorsSafeImage(clientImg);
                const canvas = Preprocessor.process(safeImg);
                const detection = await this.detectWithFallback(canvas, "foto do cliente");
                if (!detection) {
                    StatusPanel.update("Nenhum rosto detectado na foto do cliente.", true);
                    return;
                }
                this.clientEmbedding = EmbeddingGenerator.extractEmbedding(detection);
                StatusPanel.update("Embedding do cliente calculado com sucesso.");
            } catch (err) {
                StatusPanel.update(`Erro ao processar foto do cliente: ${err.message}`, true);
            }
        },

        async scanAndQueueNegativeImages() {
            StatusPanel.update(`Procurando fotos da lista ("${CONFIG.NEGATIVE_SELECTOR}")...`);
            const firstImg = await Utils.waitForElement(CONFIG.NEGATIVE_SELECTOR);

            const imgs = document.querySelectorAll(CONFIG.NEGATIVE_SELECTOR);
            if (!firstImg || imgs.length === 0) {
                StatusPanel.update(
                    `Nenhuma imagem encontrada com o seletor "${CONFIG.NEGATIVE_SELECTOR}". Verifique o seletor.`,
                    true
                );
                return;
            }

            StatusPanel.update(`${imgs.length} imagem(ns) encontrada(s) na lista. Processando...`);
            imgs.forEach((img) => this.enqueue(img));
            await this.processQueue();
            StatusPanel.update("Lote inicial de imagens processado.");

            if (CONFIG.DEBUG_PAIRWISE_ANALYSIS) {
                this.runPairwiseAnalysis();
            }
        },

        /**
         * Compara todas as fotos da lista ENTRE SI (não com o cliente) e
         * mostra no console os pares mais parecidos. O mesmo fraudador
         * pode aparecer várias vezes na lista (ângulos/dias diferentes) —
         * pares com cosseno bem alto são candidatos fortes a "mesma
         * pessoa, foto diferente", dando dados reais de calibração sem
         * precisar de um caso de cliente batendo com fraudador.
         */
        runPairwiseAnalysis() {
            const items = this.collectedEmbeddings;
            const pairs = [];

            for (let i = 0; i < items.length; i++) {
                for (let j = i + 1; j < items.length; j++) {
                    const cosine = SimilarityCalculator.cosineSimilarity(
                        items[i].embedding,
                        items[j].embedding
                    );
                    pairs.push({ cosine, srcA: items[i].src, srcB: items[j].src });
                }
            }

            pairs.sort((a, b) => b.cosine - a.cosine);

            const top = pairs.slice(0, 20).map((p) => ({
                cosine: p.cosine.toFixed(4),
                fotoA: p.srcA,
                fotoB: p.srcB,
            }));

            Logger.log(
                `Análise de pares: ${items.length} fotos, ${pairs.length} pares comparados. Top 20 mais parecidos (candidatos a "mesma pessoa"):`
            );
            console.table(top);
        },

        enqueue(imgEl) {
            if (this.processedImgs.has(imgEl)) return;
            this.processedImgs.add(imgEl);
            this.queue.push(imgEl);
        },

        async processQueue() {
            if (this.processing) return;
            this.processing = true;

            while (this.queue.length > 0) {
                const batch = this.queue.splice(0, CONFIG.BATCH_SIZE);
                await Promise.all(batch.map((img) => this.processImage(img)));

                // Cede o controle ao navegador entre lotes. Usamos só
                // setTimeout (nunca requestAnimationFrame) porque o rAF
                // simplesmente não dispara com a aba em segundo plano —
                // era isso que travava tudo quando você trocava de aba.
                // setTimeout continua rodando em background (o Chrome só
                // passa a espaçar mais os timers depois de alguns minutos
                // de aba oculta, mas não para).
                await new Promise((resolve) => setTimeout(resolve, CONFIG.BATCH_DELAY_MS));
            }

            this.processing = false;
        },

        /**
         * Retorna o embedding de uma foto da lista, usando o cache
         * persistente (GM_setValue/GM_getValue) quando disponível — assim,
         * fotos de fraudadores já vistas em consultas anteriores não
         * precisam ser baixadas nem reprocessadas de novo.
         *
         * Retorna null se nenhum rosto foi detectado (também cacheado,
         * pra não tentar detectar de novo em fotos sem rosto).
         */
        async getEmbeddingForListImage(imgEl) {
            const src = imgEl.currentSrc || imgEl.src;
            const cacheKey = `facematch_embed_v1:${src}`;

            const cached = await PersistentCache.get(cacheKey);
            if (cached === "NO_FACE") return null;
            if (cached) {
                const embedding = new Float32Array(cached);
                if (CONFIG.DEBUG_PAIRWISE_ANALYSIS) {
                    this.collectedEmbeddings.push({ src, embedding });
                }
                return embedding;
            }

            const safeImg = await this.getCorsSafeImage(imgEl);
            const canvas = Preprocessor.process(safeImg);
            const detection = await this.detectWithFallback(canvas, src);

            if (!detection) {
                await PersistentCache.set(cacheKey, "NO_FACE");
                return null;
            }

            const embedding = EmbeddingGenerator.extractEmbedding(detection);
            await PersistentCache.set(cacheKey, Array.from(embedding));

            if (CONFIG.DEBUG_PAIRWISE_ANALYSIS) {
                this.collectedEmbeddings.push({ src, embedding });
            }

            return embedding;
        },

        async processImage(imgEl) {
            if (!this.clientEmbedding) {
                Logger.warn("Embedding do cliente indisponível; pulando comparação.");
                return;
            }

            try {
                const embedding = await this.getEmbeddingForListImage(imgEl);
                if (!embedding) {
                    Overlay.showNoFace(imgEl);
                    return;
                }

                const cosine = SimilarityCalculator.cosineSimilarity(this.clientEmbedding, embedding);
                const pct = SimilarityCalculator.toPercentage(cosine);

                Overlay.showSimilarity(imgEl, pct);
                Logger.log(`Imagem processada: ${pct.toFixed(1)}% de similaridade.`);
            } catch (err) {
                Logger.error("Erro ao processar imagem da lista:", err);
                Overlay.showError(imgEl);
            }
        },

        observeMutations() {
            const observer = new MutationObserver((mutations) => {
                let foundNew = false;

                for (const mutation of mutations) {
                    mutation.addedNodes.forEach((node) => {
                        if (!(node instanceof HTMLElement)) return;

                        // Novas imagens da lista negativa adicionadas diretamente
                        if (node.matches && node.matches(CONFIG.NEGATIVE_SELECTOR)) {
                            this.enqueue(node);
                            foundNew = true;
                        }
                        // Ou aninhadas dentro do nó adicionado
                        if (node.querySelectorAll) {
                            node.querySelectorAll(CONFIG.NEGATIVE_SELECTOR).forEach((img) => {
                                this.enqueue(img);
                                foundNew = true;
                            });
                        }
                    });
                }

                if (foundNew) this.processQueue();
            });

            observer.observe(document.body, { childList: true, subtree: true });
            Logger.log("MutationObserver ativo para novas imagens.");
        },
    };

    // ------------------------------------------------------------------
    // BOOTSTRAP
    // ------------------------------------------------------------------
    if (CONFIG.AUTO_START) {
        (async () => {
            const isListaNegativa = await PageGuard.waitForMatch();

            if (!isListaNegativa) {
                Logger.log(
                    "Página atual não parece ser a Lista Negativa (título/elemento não encontrados) — script permanece ocioso."
                );
                return;
            }

            Logger.log("Tela de Lista Negativa detectada — iniciando FaceMatch.");
            Main.init().catch((err) => Logger.error("Falha na inicialização:", err));
        })();
    }

    // Exposto para depuração manual no console, se necessário.
    window.__FaceMatch = { CONFIG, Main, SimilarityCalculator, Overlay };
})();