#Method based on values used with Khazan Ansat helicopter drop test, intrapolated to fit our model
#Source: https://dspace-erf.nlr.nl/server/api/core/bitstreams/a0a7be04-c625-476c-997e-25f5dc2b40a6/content


mtow_ansat = 3,600
def size(max_elastic_stress, young_mod, mtow_kg):

    scale_factor = mtow_kg/mtow_ansat
         

