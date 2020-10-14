#### Join UCDP conflict deaths data & data_full from paper to create master dataset ####

### Set up
setwd("Data")
# Packages
pkgs <- c("tidyverse", "randomForest", "caret", "ROCR", "stepPlr", "doMC", "xtable", 
          "separationplot", "countrycode")
lapply(pkgs, require, character.only = T) ; rm(pkgs)

## Load data

# UCDP
UCDP <- read_csv("state_violence/conflict_deaths_cleaned.csv")[, -1]

# data_full

Data_full <- read_csv("data_full.csv")
Data_full <- mutate(Data_full, cowcode = as.character(cowcode))

Data_full_imp <- read_csv("data_full_imp.csv")[, -c(1:2)]
Data_full_imp <- mutate(Data_full_imp, cowcode = as.character(cowcode))

### Cleaning

# Dedup UCDP data & clean

UCDP_dedup <- group_by(UCDP, loc_id, year) %>% 
  summarise(per_capita_deaths = sum(per_capita_deaths)) %>% 
  filter(year <= 2014)
names(UCDP_dedup)[1] <- "iso3n"

# Countrycode dataframe 

Countrycode <- codelist_panel %>%
  filter(year >= 1989 & year <= 2014) %>% 
  dplyr::select(country.name.en, year, iso3n, cown)
  
# UCDP filled & w country codes

UCDP_full <- full_join(Countrycode, UCDP_dedup) %>% 
  filter(!(is.na(iso3n) & is.na(cown)))

# Fill in missing iso3n & cown codes

UCDP_full <- mutate(UCDP_full, iso3n = 
                      case_when(is.na(iso3n) ~ countrycode(cown, "cown", "iso3n"), 
                                !is.na(iso3n) ~ iso3n)) %>% 
  mutate(cown = as.character(case_when(is.na(cown) ~ countrycode(iso3n, "iso3n", "cown"), 
                          !is.na(cown) ~ cown)))

# Fix Serbia/Yugoslavia issue

UCDP_full <- filter(UCDP_full, !country.name.en == "Yugoslavia")

UCDP_full$cown[UCDP_full$country.name.en == "Serbia"] <- 345

# Drop cases w no cowcode

UCDP_full <- filter(UCDP_full, !is.na(cown))

# Fill in 0s

UCDP_full$per_capita_deaths[is.na(UCDP_full$per_capita_deaths)] <- 0

# Rename cown & drop iso3n

names(UCDP_full)[4] <- "cowcode"
UCDP_full <- dplyr::select(UCDP_full, -iso3n)

### Merge

# Merge w data_full (not imputed)
Data_full_ucdp <- left_join(UCDP_full, Data_full) %>% 
  filter(complete.cases(.))

# Merge w data_full (imputed)
Data_full_imp_ucdp <- left_join(UCDP_full, Data_full_imp) %>% 
  filter(complete.cases(.))

### Write files

write_csv(Data_full_ucdp, "master_1989_2000.csv")
write_csv(Data_full_imp_ucdp, "master_imputed.csv")
