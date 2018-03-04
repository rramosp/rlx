import numpy as np
import pandas as pd

def read_cat(fname):
    with open(fname) as f:
        lines = np.r_[f.readlines()]
    print "parsing type 11"
    t11 = pd.DataFrame([parse_tipo11(i) for i in lines if i[:2]=="11"])
    print "parsing type 13"
    t13 = pd.DataFrame([parse_tipo13(i) for i in lines if i[:2]=="13"])
    print "parsing type 14"
    t14 = pd.DataFrame([parse_tipo14(i) for i in lines if i[:2]=="14"])
    print "parsing type 15"
    t15 = pd.DataFrame([parse_tipo15(i) for i in lines if i[:2]=="15"])
    print "parsing type 17"
    t17 = pd.DataFrame([parse_tipo17(i) for i in lines if i[:2]=="17"])

    t = np.r_[[i[:2] for i in lines]]
    summary = pd.Series(t).value_counts()
    p = pd.Series(["finca", "unidad constructiva", "construccion", "bien inmueble", "cultivos agrarios", "marca inicio", "marca fin" ],
               index=["11", "13", "14", "15", "17", "01", "90"] )
    summary = pd.DataFrame([p,summary], index=["nombre", "cantidad"]).T
    summary.index.name = "tipo"

    t16 = None

    return summary, t11, t13, t14, t15, t16, t17

# registro finca
def parse_tipo11(i):
    tipo = i[:2]
    delmeh = i[23:25].strip()
    municipio = i[25:28].strip()
    parcela   = i[30:44].strip()
    cprov     = i[50:52].strip()
    nprov     = i[52:77].strip()
    cmun      = i[77:80].strip()
    cmunine   = i[80:83].strip()
    nmun      = i[83:123].strip()
    ent_menor = i[123:153].strip()
    cvia      = i[153:158].strip()
    tvia      = i[158:163].strip()
    nvia      = i[163:188].strip()
    npolice1  = i[188:192].strip()
    letra1    = i[192:193].strip()
    npolice2  = i[193:197].strip()
    letra2    = i[197:198].strip()
    km        = i[198:203].strip()
    bloque    = i[203:207].strip()
    direccion = i[215:240].strip()
    zipcode   = i[240:245].strip()
    distrito  = i[245:247].strip()
    munagreg  = i[247:250].strip()
    zonconcen = i[250:252].strip()
    codpolig  = i[252:255].strip()
    codparcela= i[255:260].strip()
    codparaje = i[260:265].strip()
    nomparaje = i[265:295].strip()

    sup      = int(i[295:305].strip())
    sup_construida_total   = int(i[305:312].strip())
    sup_construida_sobre_rasante = int(i[312:319].strip())
    sup_construida_bajo_rasante = int(i[319:326].strip())
    sup_cubierta = int(i[326:333].strip())
    x            = i[333:342]
    y            = i[342:352]
    refbice      = i[581:601].strip()
    denbice      = i[601:666].strip()


    r = [tipo, delmeh, municipio, parcela, cprov, nprov, cmun, cmunine, nmun, ent_menor, cvia, tvia, nvia,
         npolice1, letra1, npolice2, letra2, km, bloque, direccion, zipcode, distrito, munagreg,
         zonconcen, codpolig, codparcela, codparaje, nomparaje,
         sup, sup_construida_total, sup_construida_sobre_rasante, sup_construida_bajo_rasante,
         x, y, refbice, denbice]
    n = ["tipo", "delmeh", "municipio", "parcela", "cprov", "nprov", "cmun", "cmunine", "nmun", "ent_menor", "cvia", "tvia", "nvia",
         "npolice1", "letra1", "npolice2", "letra2", "km", "bloque", "direccion", "zipcode", "distrito", "munagreg",
         "zonconcen", "codpolig", "codparcela", "codparaje", "nomparaje",
         "sup", "sup_construida_total", "sup_construida_sobre_rasante", "sup_construida_bajo_rasante",
         "x", "y", "refbice", "denbice"]
    return pd.Series(r,index=n)

# registro unidad constructiva
def parse_tipo13(i):
    tipo = i[:2]
    delmeh = i[23:25].strip()
    municipio = i[25:28].strip()
    parcela   = i[30:44].strip()
    cprov     = i[50:52].strip()
    nprov     = i[52:77].strip()
    cmun      = i[77:80].strip()
    cmunine   = i[80:83].strip()
    nmun      = i[83:123].strip()
    ent_menor = i[123:153].strip()
    cvia      = i[153:158].strip()
    tvia      = i[158:163].strip()
    nvia      = i[163:188].strip()
    npolice1  = i[188:192].strip()
    letra1    = i[192:193].strip()
    npolice2  = i[193:197].strip()
    letra2    = i[197:198].strip()
    km        = i[198:203].strip()

    direccion = i[215:240].strip()

    anoconstruc = i[295:299]
    anoexact    = i[299:300]
    superficie  = int(i[300:307])
    facadelong  = int(i[307:312])

    codmatriz   = i[409:413]


    r = [tipo, delmeh, municipio, parcela, cprov, nprov, cmun, cmunine, nmun, ent_menor, cvia, tvia, nvia,
         npolice1, letra1, npolice2, letra2, km,
         direccion, anoconstruc,anoexact, superficie, facadelong, codmatriz]
    n = ["tipo", "delmeh", "municipio", "parcela", "cprov", "nprov", "cmun", "cmunine", "nmun", "ent_menor", "cvia", "tvia", "nvia",
         "npolice1", "letra1", "npolice2", "letra2", "km",
         "direccion", "anoconstruc","anoexact", "superficie", "facadelong", "codmatrix"]
    return pd.Series(r,index=n)

# registro construccion
def parse_tipo14(i):
    tipo = i[:2]

    delmeh = i[23:25].strip()
    municipio = i[25:28].strip()
    parcela   = i[30:44].strip()

    norden_construct  = i[44::48]
    norden_inmueble   = i[50:54]
    cod_construct     = i[54:58]

    bloque            = i[58:62]
    escalera          = i[62:64]
    planta            = i[64:67]
    puerta            = i[67:70]

    cdest             = i[70:73]
    tipo_reforma      = i[73:74]
    ano_reforma       = i[74:78]
    ano_antiguedad    = i[78:82]
    local_interior    = i[82:83]

    superficie_total  = int(i[83:90])
    superficie_terrazas = int(i[90:97])
    superficie_plantas  = int(i[97:104])

    tipo_construct      = i[104:109]
    cod_reparto         = i[111:114]

    r = [tipo, delmeh, municipio, parcela, norden_construct, norden_inmueble, cod_construct,
         bloque, escalera, planta, puerta, cdest, tipo_reforma, ano_reforma, ano_antiguedad, local_interior,
         superficie_total, superficie_terrazas, superficie_plantas, tipo_construct, cod_reparto]

    n = ["tipo", "delmeh", "municipio", "parcela", "norden_construct", "norden_inmueble", "cod_construct",
         "bloque", "escalera", "planta", "puerta", "cdest", "tipo_reforma", "ano_reforma", "ano_antiguedad", "local_interior",
         "superficie_total", "superficie_terrazas", "superficie_plantas", "tipo_construct", "cod_reparto"]
    return pd.Series(r,index=n)


# registro bien inmueble
def parse_tipo15(i):

    tipo = i[:2]
    delmeh = i[23:25].strip()
    municipio = i[25:28].strip()
    parcela   = i[30:44].strip()
    nseq      = i[44:48]
    control1  = i[48:49]
    control2  = i[49:50]

    nfijo     = i[50:58]
    ayunt_id  = i[58:73]
    nfinca    = i[73:92]

    cprov     = i[92:94]
    nprov     = i[94:119]

    cmun      = i[119:122].strip()
    cmunine   = i[122:125].strip()
    nmun      = i[125:165].strip()
    ent_menor = i[165:195].strip()
    cvia      = i[195:200].strip()
    tvia      = i[200:205].strip()
    nvia      = i[205:230].strip()
    npolice1  = i[230:234].strip()
    letra1    = i[234:235].strip()
    npolice2  = i[235:239].strip()
    letra2    = i[239:240].strip()
    km        = i[240:245].strip()
    bloque    = i[245:249].strip()
    escalera  = i[249:251]
    planta    = i[251:254]
    puerta    = i[254:257]
    direccion = i[257:282].strip()
    zip       = i[282:287]
    distrito  = i[287:289]

    munagreg  = i[289:293].strip()
    zonconcen = i[292:294].strip()
    codpolig  = i[294:297].strip()
    codparcela= i[297:302].strip()
    codparaje = i[302:307].strip()
    nomparaje = i[307:337].strip()

    numorden  = i[367:371].strip()
    antiguedad = i[371:375].strip()

    cgrupo     = i[427:428]
    supercifie_contruct = int(i[441:451])
    supercifie_inmueble = int(i[451:461])
    cof_propiedad = i[461:470]

    r =[tipo, delmeh, municipio, parcela, nseq, control1, control2,
        nfijo, ayunt_id, nfinca, cprov, nprov, cmun, cmunine, nmun, ent_menor,
        cvia, tvia, nvia,npolice1, letra1, npolice2, letra2, km,
        bloque, escalera, planta, puerta, direccion, zip, distrito,
        munagreg, zonconcen, codpolig, codparcela, codparaje, nomparaje,
        numorden, antiguedad, cgrupo, supercifie_contruct, supercifie_inmueble, cof_propiedad]

    n =["tipo", "delmeh", "municipio", "parcela", "nseq", "control1", "control2",
        "nfijo", "ayunt_id", "nfinca", "cprov", "nprov", "cmun", "cmunine", "nmun", "ent_menor",
        "cvia", "tvia", "nvia","npolice1", "letra1", "npolice2", "nletra2", "km",
        "bloque", "escalera", "planta", "puerta", "direccion", "zip", "distrito",
        "munagreg", "zonconcen", "codpolig", "codparcela", "codparaje", "nomparaje",
        "numorden", "antiguedad", "cgrupo", "supercifie_contruct", "supercifie_inmueble", "cof_propiedad"]
    return pd.Series(r,index=n)

# registro cultivos agrarios
def parse_tipo17(i):

    tipo = i[:2]
    delmeh = i[23:25].strip()
    municipio = i[25:28].strip()
    parcela   = i[30:44].strip()
    nseq      = i[44:48]
    control1  = i[48:49]
    control2  = i[49:50]

    norden    = i[50:54]
    tipo_subparcela = i[54:55]
    superficie = int(i[55:65])
    calificacion = i[65:67]
    clase        = i[67:107]
    produccion   = i[107:109]
    reparto      = i[126:129]


    r =[tipo, delmeh, municipio, parcela, nseq, control1, control2,
        norden, tipo_subparcela, superficie, calificacion,clase, produccion, reparto ]

    n =["tipo", "delmeh", "municipio", "parcela", "nseq", "control1", "control2",
        "norden", "tipo_subparcela", "superficie", "calificacion","clase", "produccion", "reparto" ]
    return pd.Series(r,index=n)
