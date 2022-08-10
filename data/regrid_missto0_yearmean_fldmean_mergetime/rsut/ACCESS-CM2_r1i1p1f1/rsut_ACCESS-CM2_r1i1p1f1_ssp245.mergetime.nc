CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-11-08T10:09:32Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      	�Wed Aug 10 15:18:27 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp245.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_ssp245_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_ssp245.mergetime.nc
Wed Aug 10 15:18:26 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 08:53:13 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 08:53:09 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-11-08T10:09:32Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: CM2-historical; Local ID: bj594; Variable: rsut (['fld_s01i208'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0950-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   version       	v20191108      cmor_version      3.4.0      tracking_id       1hdl:21.14100/ae0d2880-10cc-4234-90ec-f56309c77f6a      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   cell_measures         area: areacella    history       u2019-11-08T10:09:31Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   B�2�Aq�6�   Aq�P   Aq��P   B�Aq���   Aq��P   Aq��P   B�'AAq��   Aq��P   Aq�dP   B��!Aq���   Aq�dP   Aq��P   BÄ�Aq���   Aq��P   Aq�FP   B�
Aq�k�   Aq�FP   Aq��P   B�8�Aq���   Aq��P   Aq�(P   BÊ�Aq�M�   Aq�(P   Aq��P   BõdAq���   Aq��P   Aq�
P   B��Aq�/�   Aq�
P   Aq�{P   BÖvAq���   Aq�{P   Aq��P   B�SAq��   Aq��P   Aq�]P   B�Z�AqĂ�   Aq�]P   Aq��P   B���Aq���   Aq��P   Aq�?P   B�,tAq�d�   Aq�?P   Aq˰P   B�n�Aq���   Aq˰P   Aq�!P   B�5Aq�F�   Aq�!P   AqВP   Bā�Aqз�   AqВP   Aq�P   B�ýAq�(�   Aq�P   Aq�tP   B�Aqՙ�   Aq�tP   Aq��P   B��%Aq�
�   Aq��P   Aq�VP   B�9�Aq�{�   Aq�VP   Aq��P   B��Aq���   Aq��P   Aq�8P   B�kAq�]�   Aq�8P   Aq�P   B�"�Aq���   Aq�P   Aq�P   B�O8Aq�?�   Aq�P   Aq�P   BË�Aq��   Aq�P   Aq��P   B�F�Aq�!�   Aq��P   Aq�mP   B��LAq��   Aq�mP   Aq��P   B�´Aq��   Aq��P   Aq�OP   B��Aq�t�   Aq�OP   Aq��P   B�\Aq���   Aq��P   Aq�1P   B�pAq�V�   Aq�1P   Aq��P   Bǭ<Aq���   Aq��P   Aq�P   Bˀ�Aq�8�   Aq�P   Aq��P   B�"pAq���   Aq��P   Aq��P   B��%Aq��   Aq��P   ArfP   B��7Ar��   ArfP   Ar�P   B���Ar��   Ar�P   ArHP   B�6PArm�   ArHP   Ar�P   B�>�Ar��   Ar�P   Ar*P   B�6ArO�   Ar*P   Ar�P   BŀuAr��   Ar�P   ArP   Bķ�Ar1�   ArP   Ar}P   B���Ar��   Ar}P   Ar�P   B��XAr�   Ar�P   Ar_P   B��~Ar��   Ar_P   Ar�P   B�pAr��   Ar�P   ArAP   B�J�Arf�   ArAP   Ar�P   B�׹Ar��   Ar�P   Ar!#P   Bö�Ar!H�   Ar!#P   Ar#�P   B�Ar#��   Ar#�P   Ar&P   B���Ar&*�   Ar&P   Ar(vP   BǛ�Ar(��   Ar(vP   Ar*�P   B�M6Ar+�   Ar*�P   Ar-XP   B�ղAr-}�   Ar-XP   Ar/�P   B���Ar/��   Ar/�P   Ar2:P   B�t�Ar2_�   Ar2:P   Ar4�P   B�7
Ar4��   Ar4�P   Ar7P   B�+lAr7A�   Ar7P   Ar9�P   BĔ�Ar9��   Ar9�P   Ar;�P   B�ΊAr<#�   Ar;�P   Ar>oP   B�ApAr>��   Ar>oP   Ar@�P   BƸ�ArA�   Ar@�P   ArCQP   BƇ,ArCv�   ArCQP   ArE�P   B�`�ArE��   ArE�P   ArH3P   B�u�ArHX�   ArH3P   ArJ�P   BſKArJ��   ArJ�P   ArMP   B��ZArM:�   ArMP   ArO�P   B�ntArO��   ArO�P   ArQ�P   B�\�ArR�   ArQ�P   ArThP   B�c�ArT��   ArThP   ArV�P   B�SQArV��   ArV�P   ArYJP   B�:RArYo�   ArYJP   Ar[�P   B��3Ar[��   Ar[�P   Ar^,P   B��KAr^Q�   Ar^,P   Ar`�P   B�=�Ar`��   Ar`�P   ArcP   B�[�Arc3�   ArcP   AreP   B�G.Are��   AreP   Arg�P   B�FArh�   Arg�P   ArjaP   B�Arj��   ArjaP   Arl�P   B�ݽArl��   Arl�P   AroCP   B�
�Aroh�   AroCP   Arq�P   BýaArq��   Arq�P   Art%P   B�ڻArtJ�   Art%P   Arv�P   B�%�Arv��   Arv�P   AryP   BďBAry,�   AryP   Ar{xP   BįoAr{��   Ar{xP   Ar}�P   B�aAr~�   Ar}�P   Ar�ZP   B��?Ar��   Ar�ZP   Ar��P   B�CAr���   Ar��P   Ar�<P   B�a*Ar�a�   Ar�<P   Ar��P   B�)Ar���   Ar��P   Ar�P   BīZAr�C�   Ar�P   Ar��P   B��Ar���   Ar��P   Ar� P   BÊ�Ar�%�   Ar� P   Ar�qP   B�M�Ar���   Ar�qP   Ar��P   B��Ar��   Ar��P   Ar�SP   B�2}Ar�x�   Ar�SP   Ar��P   B�=JAr���   Ar��P   Ar�5P   B�9#Ar�Z�   Ar�5P   Ar��P   Bĳ3Ar���   Ar��P   Ar�P   Bĺ�Ar�<�   Ar�P   Ar��P   B�n)Ar���   Ar��P   Ar��P   B�y�Ar��   Ar��P   Ar�jP   B�1�Ar���   Ar�jP   Ar��P   B�t@Ar� �   Ar��P   Ar�LP   B�"�Ar�q�   Ar�LP   Ar��P   B�E�Ar���   Ar��P   Ar�.P   B��}Ar�S�   Ar�.P   Ar��P   B�88Ar���   Ar��P   Ar�P   B��{Ar�5�   Ar�P   Ar��P   B�Y=Ar���   Ar��P   Ar��P   B���Ar��   Ar��P   Ar�cP   BȉQAr���   Ar�cP   Ar��P   B���Ar���   Ar��P   Ar�EP   BƁAr�j�   Ar�EP   ArĶP   B��.Ar���   ArĶP   Ar�'P   B�RkAr�L�   Ar�'P   ArɘP   BƁArɽ�   ArɘP   Ar�	P   BƙZAr�.�   Ar�	P   Ar�zP   B�f3ArΟ�   Ar�zP   Ar��P   B� �Ar��   Ar��P   Ar�\P   B���ArӁ�   Ar�\P   Ar��P   BƩ�Ar���   Ar��P   Ar�>P   B�a�Ar�c�   Ar�>P   ArگP   Bƹ3Ar���   ArگP   Ar� P   B��Ar�E�   Ar� P   ArߑP   B��IAr߶�   ArߑP   Ar�P   B�G�Ar�'�   Ar�P   Ar�sP   B�N7Ar��   Ar�sP   Ar��P   B�_�Ar�	�   Ar��P   Ar�UP   B��Ar�z�   Ar�UP   Ar��P   B�<>Ar���   Ar��P   Ar�7P   B�=7Ar�\�   Ar�7P   Ar�P   B��VAr���   Ar�P   Ar�P   B��Ar�>�   Ar�P   Ar��P   Bƕ<Ar���   Ar��P   Ar��P   B��Ar� �   Ar��P   Ar�lP   B�l[Ar���   Ar�lP   Ar��P   B�VAr��   Ar��P   Ar�NP   B��Ar�s�   Ar�NP   As�P   B̅�As��   As�P   As0P   B�ǀAsU�   As0P   As�P   BƟ�As��   As�P   As	P   Bű�As	7�   As	P   As�P   B��As��   As�P   As�P   BŷzAs�   As�P   AseP   B��As��   AseP   As�P   B�EAs��   As�P   AsGP   B�tAsl�   AsGP   As�P   B�As��   As�P   As)P   BĆ:AsN�   As)P   As�P   B�`�As��   As�P   AsP   B�ɇAs0�   AsP   As!|P   B��:As!��   As!|P   As#�P   B���As$�   As#�P   As&^P   B�� As&��   As&^P   As(�P   B�7�As(��   As(�P   As+@P   BĦ�As+e�   As+@P   As-�P   BÏLAs-��   As-�P   As0"P   B��2As0G�   As0"P   As2�P   BÝ�As2��   As2�P   As5P   B��As5)�   As5P   As7uP   B���As7��   As7uP   As9�P   B���As:�   As9�P   As<WP   B�մAs<|�   As<WP   As>�P   B�M,As>��   As>�P   AsA9P   B�T�AsA^�   AsA9P   AsC�P   B�v�AsC��   AsC�P   AsFP   B�\;AsF@�   AsFP   AsH�P   B��;AsH��   AsH�P   AsJ�P   B�O�AsK"�   AsJ�P   AsMnP   B�
�AsM��   AsMnP   AsO�P   B���AsP�   AsO�P   AsRPP   B��AsRu�   AsRPP   AsT�P   B�e�AsT��   AsT�P   AsW2P   B�]!AsWW�   AsW2P   AsY�P   B�{�AsY��   AsY�P   As\P   B���As\9�   As\P   As^�P   B���As^��   As^�P   As`�P   B�F"Asa�   As`�P   AscgP   B��~Asc��   AscgP   Ase�P   B�
Ase��   Ase�P   AshIP   B���Ashn�   AshIP   Asj�P   B�V.Asj��   Asj�P   Asm+P   B�!AsmP�   Asm+P   Aso�P   B��rAso��   Aso�P   AsrP   B��(Asr2�   AsrP   Ast~P   B���Ast��   Ast~P   Asv�P   B��Asw�   Asv�P   Asy`P   B��)Asy��   Asy`P   As{�P   B��pAs{��   As{�P   As~BP   B��6As~g�   As~BP   As��P   B��gAs���   As��P   As�$P   B��MAs�I�   As�$P   As��P   B�G�As���   As��P   As�P   B�2�As�+�   As�P   As�wP   B�F�As���   As�wP   As��P   B���As��   As��P   As�YP   B�W�As�~�   As�YP   As��P   B��As���   As��P   As�;P   B�C�As�`�   As�;P   As��P   B�1qAs���   As��P   As�P   B�'GAs�B�   As�P   As��P   B�H�As���   As��P   As��P   B��)As�$�   As��P   As�pP   B�C�As���   As�pP   As��P   B���As��   As��P   As�RP   B�`>As�w�   As�RP   As��P   B�\nAs���   As��P   As�4P   B�v[As�Y�   As�4P   As��P   B��	As���   As��P   As�P   B�A�As�;�   As�P   As��P   B���As���   As��P   As��P   B�/�As��   As��P   As�iP   B���As���   As�iP   As��P   B��^As���   As��P   As�KP   B���As�p�   As�KP   As��P   B�G>As���   As��P   As�-P   B���As�R�   As�-P   AsP   B�D�As���   AsP   As�P   B��^As�4�   As�P   AsǀP   B�y�Asǥ�   AsǀP   As��P   B��PAs��   As��P   As�bP   B�*UAṡ�   As�bP   As��P   B���As���   As��P   As�DP   B��oAs�i�   As�DP   AsӵP   B��{As���   AsӵP   As�&P   B��As�K�   As�&P   AsؗP   B��Asؼ�   AsؗP   As�P   B�As�-�   As�P   As�yP   B�oAsݞ�   As�yP   As��P   B���As��   As��P   As�[P   B��RAs��   As�[P   As��P   B���As���   As��P   As�=P   B�ckAs�b�   As�=P   As�P   B��,As���   As�P   As�P   B�^As�D�   As�P   As�P   B�,As��   As�P   As�P   B�|As�&�   As�P   As�rP   B��As��   As�rP   As��P   B��As��   As��P   As�TP   B��eAs�y�   As�TP   As��P   B�|As���   As��P   As�6P   B�As�[�   As�6P   As��P   B��As���   As��P   AtP   B�׬At=�   AtP   At�P   B��At��   At�P   At�P   B��YAt�   At�P   At	kP   B���