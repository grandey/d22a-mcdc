CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A'�       creation_date         2019-11-12T01:26:23Z   data_specs_version        01.00.30   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Thttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.piControl.none.r1i1p1f1   grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      .Wed Nov 09 18:59:47 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Amon.rsut.gn.v20191112/rsut_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.yearmean.mul.areacella_piControl_v20191112.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_piControl.mergetime.nc
Fri Nov 04 05:52:43 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Amon.rsut.gn.v20191112/rsut_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.fx.areacella.gn.v20191112/areacella_fx_ACCESS-CM2_piControl_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.piControl.r1i1p1f1.Amon.rsut.gn.v20191112/rsut_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.yearmean.mul.areacella_piControl_v20191112.fldsum.nc
2019-11-12T01:26:23Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.     initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         EExp: CM2-piControl; Local ID: bi889; Variable: rsut (['fld_s01i208'])      parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0001-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   version       	v20191112      cmor_version      3.4.0      tracking_id       1hdl:21.14100/e038ae7f-3307-4df0-9435-ce289590510c      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   cell_measures         area: areacella    history       u2019-11-12T01:26:21Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Ab�   Ab��   Ab#��   [1M4Ab#�   Ab#��   Ab(|�   [1NkAb(Ǩ   Ab(|�   Ab-^�   [1;Ab-��   Ab-^�   Ab2@�   [1)�Ab2��   Ab2@�   Ab7"�   [1hyAb7m�   Ab7"�   Ab<�   [1�)Ab<O�   Ab<�   Ab@�   [1��AbA1�   Ab@�   AbEȠ   [2B�AbF�   AbEȠ   AbJ��   [1�VAbJ��   AbJ��   AbO��   [1�AbOר   AbO��   AbTn�   [1{�AbT��   AbTn�   AbYP�   [1�AbY��   AbYP�   Ab^2�   [1��Ab^}�   Ab^2�   Abc�   [1P�Abc_�   Abc�   Abg��   [12�AbhA�   Abg��   Ablؠ   [0��Abm#�   Ablؠ   Abq��   [1V�Abr�   Abq��   Abv��   [1t�Abv�   Abv��   Ab{~�   [1��Ab{ɨ   Ab{~�   Ab�`�   [1�9Ab���   Ab�`�   Ab�B�   [1��Ab���   Ab�B�   Ab�$�   [1KAb�o�   Ab�$�   Ab��   [1c
Ab�Q�   Ab��   Ab��   [1��Ab�3�   Ab��   Ab�ʠ   [1W�Ab��   Ab�ʠ   Ab���   [1��Ab���   Ab���   Ab���   [1��Ab�٨   Ab���   Ab�p�   [197Ab���   Ab�p�   Ab�R�   [1��Ab���   Ab�R�   Ab�4�   [1w�Ab��   Ab�4�   Ab��   [1UmAb�a�   Ab��   Ab���   [1`�Ab�C�   Ab���   Ab�ڠ   [2jAb�%�   Ab�ڠ   Abļ�   [1U?Ab��   Abļ�   Abɞ�   [1GAb��   Abɞ�   Ab΀�   [1��Ab�˨   Ab΀�   Ab�b�   [0�CAbӭ�   Ab�b�   Ab�D�   [1�Ab؏�   Ab�D�   Ab�&�   [1x�Ab�q�   Ab�&�   Ab��   [1�6Ab�S�   Ab��   Ab��   [1��Ab�5�   Ab��   Ab�̠   [1tAb��   Ab�̠   Ab�   [1�QAb���   Ab�   Ab���   [1��Ab�ۨ   Ab���   Ab�r�   [0��Ab���   Ab�r�   Ab�T�   [1�DAb���   Ab�T�   Ac6�   [1�3Ac��   Ac6�   Ac	�   [0��Ac	c�   Ac	�   Ac��   [13hAcE�   Ac��   Acܠ   [1
�Ac'�   Acܠ   Ac��   [14�Ac	�   Ac��   Ac��   [1�Ac�   Ac��   Ac!��   [1!,Ac!ͨ   Ac!��   Ac&d�   [1��Ac&��   Ac&d�   Ac+F�   [2 Ac+��   Ac+F�   Ac0(�   [1��Ac0s�   Ac0(�   Ac5
�   [1�Ac5U�   Ac5
�   Ac9�   [0��Ac:7�   Ac9�   Ac>Π   [1�GAc?�   Ac>Π   AcC��   [1X�AcC��   AcC��   AcH��   [0�(AcHݨ   AcH��   AcMt�   [0�"AcM��   AcMt�   AcRV�   [0�AcR��   AcRV�   AcW8�   [1W�AcW��   AcW8�   Ac\�   [0��Ac\e�   Ac\�   Ac`��   [0��AcaG�   Ac`��   Aceޠ   [0�aAcf)�   Aceޠ   Acj��   [0��Ack�   Acj��   Aco��   [1��Aco��   Aco��   Act��   [0��ActϨ   Act��   Acyf�   [0ԇAcy��   Acyf�   Ac~H�   [0�/Ac~��   Ac~H�   Ac�*�   [0H�Ac�u�   Ac�*�   Ac��   [0�^Ac�W�   Ac��   Ac��   [0�Ac�9�   Ac��   Ac�Р   [0z�Ac��   Ac�Р   Ac���   [0ηAc���   Ac���   Ac���   [0��Ac�ߨ   Ac���   Ac�v�   [0��Ac���   Ac�v�   Ac�X�   [0��Ac���   Ac�X�   Ac�:�   [0�XAc���   Ac�:�   Ac��   [1aQAc�g�   Ac��   Ac���   [1D�Ac�I�   Ac���   Ac��   [1�$Ac�+�   Ac��   Ac�    [1(�Ac��   Ac�    Ac¤�   [0�kAc��   Ac¤�   Acǆ�   [1z�Ac�Ѩ   Acǆ�   Ac�h�   [1�Ac̳�   Ac�h�   Ac�J�   [1`HAcѕ�   Ac�J�   Ac�,�   [1��Ac�w�   Ac�,�   Ac��   [1 Ac�Y�   Ac��   Ac��   [1�FAc�;�   Ac��   Ac�Ҡ   [1vbAc��   Ac�Ҡ   Ac鴠   [1�MAc���   Ac鴠   Ac   [1�Ac��   Ac   Ac�x�   [1�Ac�è   Ac�x�   Ac�Z�   [1PYAc���   Ac�Z�   Ac�<�   [2N�Ac���   Ac�<�   Ad�   [1pAdi�   Ad�   Ad �   [1^�AdK�   Ad �   Ad�   [11Ad-�   Ad�   AdĠ   [1N�Ad�   AdĠ   Ad��   [1�Ad�   Ad��   Ad��   [1/AdӨ   Ad��   Adj�   [1KLAd��   Adj�   Ad$L�   [0�@Ad$��   Ad$L�   Ad).�   [0��Ad)y�   Ad).�   Ad.�   [1JAd.[�   Ad.�   Ad2�   [0[�Ad3=�   Ad2�   Ad7Ԡ   [1~rAd8�   Ad7Ԡ   Ad<��   [1�Ad=�   Ad<��   AdA��   [1<�AdA�   AdA��   AdFz�   [1��AdFŨ   AdFz�   AdK\�   [0�$AdK��   AdK\�   AdP>�   [0��AdP��   AdP>�   AdU �   [0��AdUk�   AdU �   AdZ�   [0�}AdZM�   AdZ�   Ad^�   [1eAd_/�   Ad^�   AdcƠ   [1/�Add�   AdcƠ   Adh��   [0ϸAdh�   Adh��   Adm��   [1D�Admը   Adm��   Adrl�   [1s(Adr��   Adrl�   AdwN�   [1}�Adw��   AdwN�   Ad|0�   [17Ad|{�   Ad|0�   Ad��   [1/�Ad�]�   Ad��   Ad���   [1˖Ad�?�   Ad���   Ad�֠   [1w�Ad�!�   Ad�֠   Ad���   [15�Ad��   Ad���   Ad���   [1%�Ad��   Ad���   Ad�|�   [1��Ad�Ǩ   Ad�|�   Ad�^�   [1�IAd���   Ad�^�   Ad�@�   [1XAd���   Ad�@�   Ad�"�   [1K�Ad�m�   Ad�"�   Ad��   [1o�Ad�O�   Ad��   Ad��   [1u�Ad�1�   Ad��   Ad�Ƞ   [1u�Ad��   Ad�Ƞ   Ad���   [2:�Ad���   Ad���   Ad���   [1�QAd�ר   Ad���   Ad�n�   [1��AdŹ�   Ad�n�   Ad�P�   [1E�Adʛ�   Ad�P�   Ad�2�   [2,�Ad�}�   Ad�2�   Ad��   [1��Ad�_�   Ad��   Ad���   [1��Ad�A�   Ad���   Ad�ؠ   [1b�Ad�#�   Ad�ؠ   Ad⺠   [1�Ad��   Ad⺠   Ad眠   [1��Ad��   Ad眠   Ad�~�   [1�Ad�ɨ   Ad�~�   Ad�`�   [0��Ad�   Ad�`�   Ad�B�   [1��Ad���   Ad�B�   Ad�$�   [1��Ad�o�   Ad�$�   Ae �   [1�Ae Q�   Ae �   Ae�   [1� Ae3�   Ae�   Ae	ʠ   [1�,Ae
�   Ae	ʠ   Ae��   [1ǞAe��   Ae��   Ae��   [0��Ae٨   Ae��   Aep�   [1i�Ae��   Aep�   AeR�   [1��Ae��   AeR�   Ae"4�   [26Ae"�   Ae"4�   Ae'�   [22yAe'a�   Ae'�   Ae+��   [1��Ae,C�   Ae+��   Ae0ڠ   [1*RAe1%�   Ae0ڠ   Ae5��   [1
Ae6�   Ae5��   Ae:��   [1NAe:�   Ae:��   Ae?��   [1��Ae?˨   Ae?��   AeDb�   [1��AeD��   AeDb�   AeID�   [1	AeI��   AeID�   AeN&�   [1N�AeNq�   AeN&�   AeS�   [1uAeSS�   AeS�   AeW�   [0�lAeX5�   AeW�   Ae\̠   [0�WAe]�   Ae\̠   Aea��   [1�Aea��   Aea��   Aef��   [0>�Aefۨ   Aef��   Aekr�   [0�XAek��   Aekr�   AepT�   [0YAep��   AepT�   Aeu6�   [10�Aeu��   Aeu6�   Aez�   [0��Aezc�   Aez�   Ae~��   [0q�AeE�   Ae~��   Ae�ܠ   [0�-Ae�'�   Ae�ܠ   Ae���   [1.�Ae�	�   Ae���   Ae���   [0�HAe��   Ae���   Ae���   [1'Ae�ͨ   Ae���   Ae�d�   [1#VAe���   Ae�d�   Ae�F�   [1�BAe���   Ae�F�   Ae�(�   [0�Ae�s�   Ae�(�   Ae�
�   [1KUAe�U�   Ae�
�   Ae��   [0�MAe�7�   Ae��   Ae�Π   [1O�Ae��   Ae�Π   Ae���   [0�Ae���   Ae���   Ae���   [1=WAe�ݨ   Ae���   Ae�t�   [0�qAe���   Ae�t�   Ae�V�   [0�Aeá�   Ae�V�   Ae�8�   [1�Aeȃ�   Ae�8�   Ae��   [0�!Ae�e�   Ae��   Ae���   [1:cAe�G�   Ae���   Ae�ޠ   [0ׂAe�)�   Ae�ޠ   Ae���   [1�Ae��   Ae���   Aeࢠ   [1vAe���   Aeࢠ   Ae儠   [1�Ae�Ϩ   Ae儠   Ae�f�   [1@�Ae걨   Ae�f�   Ae�H�   [0�!Ae   Ae�H�   Ae�*�   [0�[Ae�u�   Ae�*�   Ae��   [0n�Ae�W�   Ae��   Ae��   [1lzAe�9�   Ae��   AfР   [0�Af�   AfР   Af��   [14
Af��   Af��   Af��   [0��Afߨ   Af��   Afv�   [1�Af��   Afv�   AfX�   [1�+Af��   AfX�   Af:�   [1(�Af��   Af:�   Af �   [1�Af g�   Af �   Af$��   [1f~Af%I�   Af$��   Af)�   [1I�Af*+�   Af)�   Af.    [1�
Af/�   Af.    Af3��   [0ppAf3�   Af3��   Af8��   [1zEAf8Ѩ   Af8��   Af=h�   [1]�Af=��   Af=h�   AfBJ�   [1��AfB��   AfBJ�   AfG,�   [1]�AfGw�   AfG,�   AfL�   [0ٙAfLY�   AfL�   AfP�   [0�KAfQ;�   AfP�   AfUҠ   [0��AfV�   AfUҠ   AfZ��   [1�AfZ��   AfZ��   Af_��   [0�*Af_�   Af_��   Afdx�   [0��Afdè   Afdx�   AfiZ�   [1CAfi��   AfiZ�   Afn<�   [0�Afn��   Afn<�   Afs�   [1x�Afsi�   Afs�   Afx �   [0ӯAfxK�   Afx �   Af|�   [0�?Af}-�   Af|�   Af�Ġ   [1Z+Af��   Af�Ġ   Af���   [0ɺAf��   Af���   Af���   [1%�Af�Ө   Af���   Af�j�   [0��Af���   Af�j�   Af�L�   [0�fAf���   Af�L�   Af�.�   [1 �Af�y�   Af�.�   Af��   [1�Af�[�   Af��   Af��   [0�~Af�=�   Af��   Af�Ԡ   [0��Af��   Af�Ԡ   Af���   [19MAf��   Af���   Af���   [0�rAf��   Af���   Af�z�   [0�YAf�Ũ   Af�z�   Af�\�   [1N.Af���   Af�\�   Af�>�   [0��Af���   Af�>�   Af� �   [11Af�k�   Af� �   Af��   [1 Af�M�   Af��   Af��   [1�oAf�/�   Af��   Af�Ơ   [1'*Af��   Af�Ơ   Af٨�   [086Af��   Af٨�   Afފ�   [0��Af�ը   Afފ�   Af�l�   [1�Af㷨   Af�l�   Af�N�   [1\�Af虨   Af�N�   Af�0�   [1{�Af�{�   Af�0�   Af��   [14~Af�]�   Af��   Af���   [1U�Af�?�   Af���   Af�֠   [1dJAf�!�   Af�֠   Ag ��   [1SWAg�   Ag ��   Ag��   [1"0Ag�   Ag��   Ag
|�   [1!�Ag
Ǩ   Ag
|�   Ag^�   [0��Ag��   Ag^�   Ag@�   [0�hAg��   Ag@�   Ag"�   [0�wAgm�   Ag"�   Ag�   [0CTAgO�   Ag�   Ag"�   [0��Ag#1�   Ag"�   Ag'Ƞ   [0�!Ag(�   Ag'Ƞ   Ag,��   [19�Ag,��   Ag,��   Ag1��   [0��Ag1ר   Ag1��   Ag6n�   [1=9Ag6��   Ag6n�   Ag;P�   [1�Ag;��   Ag;P�   Ag@2�   [0�Ag@}�   Ag@2�   AgE�   [1~AgE_�   AgE�   AgI��   [0��AgJA�   AgI��   AgNؠ   [0�|AgO#�   AgNؠ   AgS��   [0@�AgT�   AgS��   AgX��   [0ޞAgX�   AgX��   Ag]~�   [0��Ag]ɨ   Ag]~�   Agb`�   [1�[Agb��   Agb`�   AggB�   [1\bAgg��   AggB�   Agl$�   [1 ^Aglo�   Agl$�   Agq�   [1|nAgqQ�   Agq�   Agu�   [1;'Agv3�   Agu�   Agzʠ   [0�SAg{�   Agzʠ   Ag��   [0�rAg��   Ag��   Ag���   [0�=Ag�٨   Ag���   Ag�p�   [1>�Ag���   Ag�p�   Ag�R�   [0goAg���   Ag�R�   Ag�4�   [0�hAg��   Ag�4�   Ag��   [0�qAg�a�   Ag��   Ag���   [13�Ag�C�   Ag���   Ag�ڠ   [0�$Ag�%�   Ag�ڠ   Ag���   [1 �Ag��   Ag���   Ag���   [0��Ag��   Ag���   Ag���   [0��Ag�˨   Ag���   Ag�b�   [0�[Ag���   Ag�b�   Ag�D�   [0�Ag���   Ag�D�   Ag�&�   [1m�Ag�q�   Ag�&�   Ag��   [1*Ag�S�   Ag��   Ag��   [0�qAg�5�   Ag��   Ag�̠   [1NAg��   Ag�̠   AgҮ�   [0�gAg���   AgҮ�   Agא�   [1��Ag�ۨ   Agא�   Ag�r�   [1h�Agܽ�   Ag�r�   Ag�T�   [1ٺAg៨   Ag�T�   Ag�6�   [1�Ag恨   Ag�6�   Ag��   [1�$Ag�c�   Ag��   Ag���   [1i�Ag�E�   Ag���   Ag�ܠ   [0��Ag�'�   Ag�ܠ   Ag���   [0��Ag�	�   Ag���   Ag���   [1Ag��   Ag���   Ah��   [0�Ahͨ   Ah��   Ahd�   [1YAh��   Ahd�   AhF�   [1E�Ah��   AhF�   Ah(�   [0��Ahs�   Ah(�   Ah
�   [1vAhU�   Ah
�   Ah�   [1J�Ah7�   Ah�   Ah Π   [1V�Ah!�   Ah Π   Ah%��   [1|�Ah%��   Ah%��   Ah*��   [1,(Ah*ݨ   Ah*��   Ah/t�   [1�Ah/��   Ah/t�   Ah4V�   [1X�Ah4��   Ah4V�   Ah98�   [0IAAh9��   Ah98�   Ah>�   [1H�Ah>e�   Ah>�   AhB��   [0��AhCG�   AhB��   AhGޠ   [0�
AhH)�   AhGޠ   AhL��   [1E&AhM�   AhL��   AhQ��   [1��AhQ��   AhQ��   AhV��   [1-�AhVϨ   AhV��   Ah[f�   [1��Ah[��   Ah[f�   Ah`H�   [1HAh`��   Ah`H�   Ahe*�   [0�Aheu�   Ahe*�   Ahj�   [0ڋAhjW�   Ahj�   Ahn�   [1QAho9�   Ahn�   AhsР   [0��Aht�   AhsР   Ahx��   [0�>Ahx��   Ahx��   Ah}��   [0ʚAh}ߨ   Ah}��   Ah�v�   [0��Ah���   Ah�v�   Ah�X�   [0I�Ah���   Ah�X�   Ah�:�   [0��Ah���   Ah�:�   Ah��   [1a�Ah�g�   Ah��   Ah���   [0�gAh�I�   Ah���   Ah��   [1%�Ah�+�   Ah��   Ah�    [16�Ah��   Ah�    Ah���   [1�Ah��   Ah���   Ah���   [0�MAh�Ѩ   Ah���   Ah�h�   [1F]Ah���   Ah�h�   Ah�J�   [0��Ah���   Ah�J�   Ah�,�   [0;~Ah�w�   Ah�,�   Ah��   [0�~Ah�Y�   Ah��   Ah��   [1n�Ah�;�   Ah��   Ah�Ҡ   [1&�Ah��   Ah�Ҡ   Ah˴�   [/�Ah���   Ah˴�   AhЖ�   [0�?Ah��   AhЖ�   Ah�x�   [05�Ah�è   Ah�x�   Ah�Z�   [1	Ahڥ�   Ah�Z�   Ah�<�   [0ÛAh߇�   Ah�<�   Ah��   [0�Ah�i�   Ah��   Ah� �   [1{Ah�K�   Ah� �   Ah��   [0p�Ah�-�   Ah��   Ah�Ġ   [0cAh��   Ah�Ġ   Ah���   [0�YAh��   Ah���   Ah���   [0߾Ah�Ө   Ah���   Aij�   [0��Ai��   Aij�   AiL�   [1_Ai��   AiL�   Ai.�   [1~	Aiy�   Ai.�   Ai�   [1k�Ai[�   Ai�   Ai�   [1$WAi=�   Ai�   AiԠ   [0�Ai�   AiԠ   Ai��   [0��Ai�   Ai��   Ai#��   [0�*Ai#�   Ai#��   Ai(z�   [0��Ai(Ũ   Ai(z�   Ai-\�   [1aAAi-��   Ai-\�   Ai2>�   [0�Ai2��   Ai2>�   Ai7 �   [0�Ai7k�   Ai7 �   Ai<�   [0�qAi<M�   Ai<�   Ai@�   [1=AiA/�   Ai@�   AiEƠ   [0��AiF�   AiEƠ   AiJ��   [0�AAiJ�   AiJ��   AiO��   [1hAiOը   AiO��   AiTl�   [0��AiT��   AiTl�   AiYN�   [0��AiY��   AiYN�   Ai^0�   [0��Ai^{�   Ai^0�   Aic�   [1.4Aic]�   Aic�   Aig��   [1��Aih?�   Aig��   Ail֠   [1|Aim!�   Ail֠   Aiq��   [1ZAir�   Aiq��   Aiv��   [1d�Aiv�   Aiv��   Ai{|�   [0��Ai{Ǩ   Ai{|�   Ai�^�   [0�Ai���   Ai�^�   Ai�@�   [0��Ai���   Ai�@�   Ai�"�   [0��Ai�m�   Ai�"�   Ai��   [1q�Ai�O�   Ai��   Ai��   [0�eAi�1�   Ai��   Ai�Ƞ   [0΅Ai��   Ai�Ƞ   Ai���   [0��Ai���   Ai���   Ai���   [0+�Ai�ר   Ai���   Ai�n�   [1�Ai���   Ai�n�   Ai�P�   [0�Ai���   Ai�P�   Ai�2�   [1:�Ai�}�   Ai�2�   Ai��   [0��Ai�_�   Ai��   Ai���   [0V�Ai�A�   Ai���   Ai�ؠ   [0[�Ai�#�   Ai�ؠ   Aiĺ�   [0��Ai��   Aiĺ�   Aiɜ�   [1��Ai��   Aiɜ�   Ai�~�   [1i�Ai�ɨ   Ai�~�   Ai�`�   [1k�Aiӫ�   Ai�`�   Ai�B�   [1ZVAi؍�   Ai�B�   Ai�$�   [1N�Ai�o�   Ai�$�   Ai��   [1d�Ai�Q�   Ai��   Ai��   [0��Ai�3�   Ai��   Ai�ʠ   [0��Ai��   Ai�ʠ   Ai�   [0�eAi���   Ai�   Ai���   [1�Ai�٨   Ai���   Ai�p�   [0��Ai���   Ai�p�   Ai�R�   [0A�Ai���   Ai�R�   Aj4�   [/��Aj�   Aj4�   Aj	�   [0N�Aj	a�   Aj	�   Aj��   [0A{AjC�   Aj��   Ajڠ   [0g�Aj%�   Ajڠ   Aj��   [0��Aj�   Aj��   Aj��   [0k=Aj�   Aj��   Aj!��   [0qBAj!˨   Aj!��   Aj&b�   [0lAj&��   Aj&b�   Aj+D�   [1Aj+��   Aj+D�   Aj0&�   [0�Aj0q�   Aj0&�   Aj5�   [19`Aj5S�   Aj5�   Aj9�   [1dAj:5�   Aj9�   Aj>̠   [0�@Aj?�   Aj>̠   AjC��   [0SAjC��   AjC��   AjH��   [0�DAjHۨ   AjH��   AjMr�   [0��AjM��   AjMr�   AjRT�   [0�AjR��   AjRT�   AjW6�   [0�fAjW��   AjW6�   Aj\�   [0�_Aj\c�   Aj\�   Aj`��   [1]�AjaE�   Aj`��   Ajeܠ   [1�Ajf'�   Ajeܠ   Ajj��   [0�Ajk	�   Ajj��   Ajo��   [0�Ajo�   Ajo��   Ajt��   [/�Ajtͨ   Ajt��   Ajyd�   [0X!Ajy��   Ajyd�   Aj~F�   [0eAj~��   Aj~F�   Aj�(�   [0Aj�s�   Aj�(�   Aj�
�   [0G�Aj�U�   Aj�
�   Aj��   [0�!Aj�7�   Aj��   Aj�Π   [0�$Aj��   Aj�Π   Aj���   [/��Aj���   Aj���   Aj���   [0b�Aj�ݨ   Aj���   Aj�t�   [0p/Aj���   Aj�t�   Aj�V�   [0�LAj���   Aj�V�   Aj�8�   [0s?Aj���   Aj�8�   Aj��   [0BAj�e�   Aj��   Aj���   [0�Aj�G�   Aj���   Aj�ޠ   [0	Aj�)�   Aj�ޠ   Aj���   [/��Aj��   Aj���   Aj¢�   [0cUAj���   Aj¢�   AjǄ�   [0?nAj�Ϩ   AjǄ�   Aj�f�   [0��Aj̱�   Aj�f�   Aj�H�   [0�JAjѓ�   Aj�H�   Aj�*�   [0��Aj�u�   Aj�*�   Aj��   [0J�Aj�W�   Aj��   Aj��   [06Aj�9�   Aj��   Aj�Р   [0]8Aj��   Aj�Р   Aj鲠   [0 �Aj���   Aj鲠   Aj   [0(�Aj�ߨ   Aj   Aj�v�   [0��Aj���   Aj�v�   Aj�X�   [0��Aj���   Aj�X�   Aj�:�   [0�Aj���   Aj�:�   Ak�   [0QgAkg�   Ak�   Ak��   [01}AkI�   Ak��   Ak�   [0]�Ak+�   Ak�   Ak    [0�Ak�   Ak    Ak��   [1*�Ak�   Ak��   Ak��   [0]AkѨ   Ak��   Akh�   [/�&Ak��   Akh�   Ak$J�   [0��Ak$��   Ak$J�   Ak),�   [0�eAk)w�   Ak),�   Ak.�   [0}�Ak.Y�   Ak.�   Ak2�   [0��Ak3;�   Ak2�   Ak7Ҡ   [0=SAk8�   Ak7Ҡ   Ak<��   [0)�Ak<��   Ak<��   AkA��   [09!AkA�   AkA��   AkFx�   [0UxAkFè   AkFx�   AkKZ�   [0<�AkK��   AkKZ�   AkP<�   [/�xAkP��   AkP<�   AkU�   [0�AkUi�   AkU�   AkZ �   [0��AkZK�   AkZ �   Ak^�   [0sAk_-�   Ak^�   AkcĠ   [/ڜAkd�   AkcĠ   Akh��   [0+~Akh�   Akh��   Akm��   [0GAkmӨ   Akm��   Akrj�   [0��Akr��   Akrj�   AkwL�   [0�_Akw��   AkwL�   Ak|.�   [0-�Ak|y�   Ak|.�   Ak��   [0�Ak�[�   Ak��   Ak��   [046Ak�=�   Ak��   Ak�Ԡ   [0)YAk��   Ak�Ԡ   Ak���   [/�]Ak��   Ak���   Ak���   [0gAk��   Ak���   Ak�z�   [0>xAk�Ũ   Ak�z�   Ak�\�   [0VAk���   Ak�\�   Ak�>�   [1�Ak���   Ak�>�   Ak� �   [1Y