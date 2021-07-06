#' @description Classificação do Extremo Oeste da Bahia
#' @author Felipe Carvalho

# ---- Carregando os pacotes ----
library(sits) # Versão v0.12.0.
library(sf)
library(dplyr)

# ---- Adicionando semente ----
set.seed(123)

# ---- Definição da quantidade de memória e nucleos que serão usados ----
n_cores <- 20
n_mem <- 40

# ---- Creating sits cube ----
message("Criação do cubo")

bdc_url <- Sys.getenv("BDC_URL_STAC")
if (is.null(bdc_url))
    bdc_url <- "https://brazildatacube.dpi.inpe.br/stac/"

cbers_cube <- sits::sits_cube(
    source = "BDC",
    url = bdc_url,
    collection = "CB4_64_16D_STK-1",
    name = "cbers_cube",
    tiles = c("022024"),
    start_date = "2018-09-01",
    end_date = "2019-08-01"
)


# ---- Extração de séries temporais ----
message("Extraindo as series temporais")
samples_path <- "./data/raw_data/samples_bdc/samples_bdc.shp"

ts <- sits::sits_get_data(cbers_cube,
                          file = samples_path,
                          shp_attr = "label",
                          multicores = 8)

saveRDS(ts, "./data/derived_data/timeseries/samples.rds")

# ---- Treinamento do modelo ----
message("Treinando o modelo")
model <- sits::sits_train(ts, ml_method = sits::sits_rfor(num_trees = 2000))

# ---- Classificação do mapa ----
message("Classificando o mapa")
probs <- sits::sits_classify(data = cbers_cube,
                             ml_model = model,
                             memsize = n_mem,
                             multicores = n_cores,
                             output_dir = "./data/derived_data/prob_cube/")

# ---- Suavização dos mapa ----
message("Suavizacao do mapa ")
smooth_bayes <- sits::sits_smooth(probs,
                                  type = "bayes",
                                  output_dir = "./data/derived_data/class/",
                                  multicores = n_mem,
                                  memsize = n_cores)
saveRDS(smooth_bayes, "./data/derived_data/class/smooth_cube.rds")

# ---- Final labeled map ----
message("Geração do mapa de classes")
maps <- sits::sits_label_classification(smooth_bayes,
                                        output_dir = "./data/derived_data/class/")
saveRDS(maps, "./data/derived_data/class/final_map.rds")