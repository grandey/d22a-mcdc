CDF   �   
      time       bnds      lon       lat          6   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-09-15T06:44:25Z   description       CMIP6 historical   title         =CNRM-ESM2-1 model output prepared for CMIP6 / CMIP historical      activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.13   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Vhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.historical.none.r1i1p1f2     grid      ldata regridded to a T127 gaussian grid (128x256 latlon) from a native atmosphere T127l reduced gaussian grid   
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         atmos      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rsut   variant_label         r1i1p1f2   EXPID         "CNRM-ESM2-1_historical_r1i1p1f2_v2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          92ddb3d0d8ce79f498d792fc8e559dcf   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      history      	�Wed Aug 10 15:18:41 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/CNRM-ESM2-1_r1i1p1f2/rsut_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.ScenarioMIP.CNRM-CERFACS.CNRM-ESM2-1.ssp126.r1i1p1f2.Amon.rsut.gr.v20190328/rsut_Amon_CNRM-ESM2-1_ssp126_r1i1p1f2_gr_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/CNRM-ESM2-1_r1i1p1f2/rsut_CNRM-ESM2-1_r1i1p1f2_ssp126.mergetime.nc
Wed Aug 10 15:18:40 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsut.gr.v20181206/rsut_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/CNRM-ESM2-1_r1i1p1f2/rsut_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc
Fri Apr 08 09:12:48 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsut.gr.v20181206/rsut_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsut.gr.v20181206/rsut_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 09:12:44 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsut.gr.v20181206/rsut_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsut.gr.v20181206/rsut_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsut.gr.v20181206/rsut_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.nc
none      tracking_id       1hdl:21.14100/f5c93cdf-1386-4771-947e-21eed25b361d      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               (   	time_bnds                                 0   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X                  lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y                   rsut                      standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       at the top of the atmosphere   positive      up     history       none   cell_measures         area: areacella             @                Aq���   Aq��P   Aq�P   B�S�Aq�6�   Aq�P   Aq��P   BƜ'Aq���   Aq��P   Aq��P   B��vAq��   Aq��P   Aq�dP   BŸ�Aq���   Aq�dP   Aq��P   B���Aq���   Aq��P   Aq�FP   BƬ�Aq�k�   Aq�FP   Aq��P   B��NAq���   Aq��P   Aq�(P   B��lAq�M�   Aq�(P   Aq��P   B�{�Aq���   Aq��P   Aq�
P   B�OAq�/�   Aq�
P   Aq�{P   B�n�Aq���   Aq�{P   Aq��P   Bƍ�Aq��   Aq��P   Aq�]P   B���AqĂ�   Aq�]P   Aq��P   B��SAq���   Aq��P   Aq�?P   B�-vAq�d�   Aq�?P   Aq˰P   BƆuAq���   Aq˰P   Aq�!P   B�,!Aq�F�   Aq�!P   AqВP   B�Aqз�   AqВP   Aq�P   BƛlAq�(�   Aq�P   Aq�tP   B��iAqՙ�   Aq�tP   Aq��P   B�_�Aq�
�   Aq��P   Aq�VP   BƬAq�{�   Aq�VP   Aq��P   BƤyAq���   Aq��P   Aq�8P   B��Aq�]�   Aq�8P   Aq�P   B�9�Aq���   Aq�P   Aq�P   B�WAq�?�   Aq�P   Aq�P   Bƪ
Aq��   Aq�P   Aq��P   B��WAq�!�   Aq��P   Aq�mP   B���Aq��   Aq�mP   Aq��P   B�h�Aq��   Aq��P   Aq�OP   BƊ�Aq�t�   Aq�OP   Aq��P   B���Aq���   Aq��P   Aq�1P   B��Aq�V�   Aq�1P   Aq��P   B�֑Aq���   Aq��P   Aq�P   B�=�Aq�8�   Aq�P   Aq��P   B��sAq���   Aq��P   Aq��P   B�=�Aq��   Aq��P   ArfP   B��:Ar��   ArfP   Ar�P   B�SJAr��   Ar�P   ArHP   B��Arm�   ArHP   Ar�P   BǄlAr��   Ar�P   Ar*P   B�5ArO�   Ar*P   Ar�P   B�QAr��   Ar�P   ArP   B�d7Ar1�   ArP   Ar}P   B�w8Ar��   Ar}P   Ar�P   B�y�Ar�   Ar�P   Ar_P   B�n�Ar��   Ar_P   Ar�P   B�oAr��   Ar�P   ArAP   B�cOArf�   ArAP   Ar�P   B�;(Ar��   Ar�P   Ar!#P   B�Ar!H�   Ar!#P   Ar#�P   B��Ar#��   Ar#�P   Ar&P   BǊ�Ar&*�   Ar&P   Ar(vP   BȪqAr(��   Ar(vP   Ar*�P   BȬ4Ar+�   Ar*�P   Ar-XP   B�K�Ar-}�   Ar-XP   Ar/�P   B���Ar/��   Ar/�P   Ar2:P   B� RAr2_�   Ar2:P   Ar4�P   B�GAr4��   Ar4�P   Ar7P   B�a(Ar7A�   Ar7P   Ar9�P   B� ZAr9��   Ar9�P   Ar;�P   B�9�Ar<#�   Ar;�P   Ar>oP   B�xcAr>��   Ar>oP   Ar@�P   B��HArA�   Ar@�P   ArCQP   B��\ArCv�   ArCQP   ArE�P   B��CArE��   ArE�P   ArH3P   B��ArHX�   ArH3P   ArJ�P   BȕSArJ��   ArJ�P   ArMP   B��ArM:�   ArMP   ArO�P   B��aArO��   ArO�P   ArQ�P   BǄ�ArR�   ArQ�P   ArThP   BǟArT��   ArThP   ArV�P   Bǯ�ArV��   ArV�P   ArYJP   BǽKArYo�   ArYJP   Ar[�P   B�
,Ar[��   Ar[�P   Ar^,P   B�TAr^Q�   Ar^,P   Ar`�P   B�oTAr`��   Ar`�P   ArcP   B�9;Arc3�   ArcP   AreP   BǂAre��   AreP   Arg�P   B�V�Arh�   Arg�P   ArjaP   B�'Arj��   ArjaP   Arl�P   B�1�Arl��   Arl�P   AroCP   B�nAroh�   AroCP   Arq�P   B���Arq��   Arq�P   Art%P   B���ArtJ�   Art%P   Arv�P   BǊ�Arv��   Arv�P   AryP   B�ؼAry,�   AryP   Ar{xP   B�6qAr{��   Ar{xP   Ar}�P   B���Ar~�   Ar}�P   Ar�ZP   B�[�Ar��   Ar�ZP   Ar��P   B��Ar���   Ar��P   Ar�<P   B�gAr�a�   Ar�<P   Ar��P   B��^Ar���   Ar��P   Ar�P   BǫTAr�C�   Ar�P   Ar��P   B� Ar���   Ar��P   Ar� P   BǓIAr�%�   Ar� P   Ar�qP   B�?IAr���   Ar�qP   Ar��P   B���Ar��   Ar��P   Ar�SP   B�שAr�x�   Ar�SP   Ar��P   Bǜ�Ar���   Ar��P   Ar�5P   B�+�Ar�Z�   Ar�5P   Ar��P   BȞ�Ar���   Ar��P   Ar�P   Bǫ�Ar�<�   Ar�P   Ar��P   B�L�Ar���   Ar��P   Ar��P   B�|TAr��   Ar��P   Ar�jP   B�
8Ar���   Ar�jP   Ar��P   B�!�Ar� �   Ar��P   Ar�LP   B���Ar�q�   Ar�LP   Ar��P   B��Ar���   Ar��P   Ar�.P   Bǵ'Ar�S�   Ar�.P   Ar��P   B��bAr���   Ar��P   Ar�P   B��BAr�5�   Ar�P   Ar��P   B�*�Ar���   Ar��P   Ar��P   B�m�Ar��   Ar��P   Ar�cP   B�3�Ar���   Ar�cP   Ar��P   B�+�Ar���   Ar��P   Ar�EP   B�-�Ar�j�   Ar�EP   ArĶP   Bȿ�Ar���   ArĶP   Ar�'P   Bǘ'Ar�L�   Ar�'P   ArɘP   BȃhArɽ�   ArɘP   Ar�	P   B�zAr�.�   Ar�	P   Ar�zP   B�#ArΟ�   Ar�zP   Ar��P   B�ѦAr��   Ar��P   Ar�\P   BȐ7ArӁ�   Ar�\P   Ar��P   B��eAr���   Ar��P   Ar�>P   B�w^Ar�c�   Ar�>P   ArگP   B�nVAr���   ArگP   Ar� P   B��BAr�E�   Ar� P   ArߑP   B��bAr߶�   ArߑP   Ar�P   B�7�Ar�'�   Ar�P   Ar�sP   B�`.Ar��   Ar�sP   Ar��P   B��wAr�	�   Ar��P   Ar�UP   B���Ar�z�   Ar�UP   Ar��P   B�!�Ar���   Ar��P   Ar�7P   B�JXAr�\�   Ar�7P   Ar�P   BȺ�Ar���   Ar�P   Ar�P   B�AAr�>�   Ar�P   Ar��P   Bǯ�Ar���   Ar��P   Ar��P   B��Ar� �   Ar��P   Ar�lP   B��7Ar���   Ar�lP   Ar��P   B��Ar��   Ar��P   Ar�NP   B�\Ar�s�   Ar�NP   As�P   B�TAs��   As�P   As0P   B��AsU�   As0P   As�P   B�hAs��   As�P   As	P   B�MhAs	7�   As	P   As�P   B�?�As��   As�P   As�P   BǾ�As�   As�P   AseP   B�5�As��   AseP   As�P   B��As��   As�P   AsGP   B�6?Asl�   AsGP   As�P   B�V�As��   As�P   As)P   B��AsN�   As)P   As�P   B���As��   As�P   AsP   Bǭ�As0�   AsP   As!|P   Bǭ�As!��   As!|P   As#�P   BǍLAs$�   As#�P   As&^P   B�]lAs&��   As&^P   As(�P   B�r�As(��   As(�P   As+@P   B�.As+e�   As+@P   As-�P   B�->As-��   As-�P   As0"P   Bǀ[As0G�   As0"P   As2�P   BǌLAs2��   As2�P   As5P   B�a!As5)�   As5P   As7uP   B�x�As7��   As7uP   As9�P   B��pAs:�   As9�P   As<WP   B�-cAs<|�   As<WP   As>�P   B��As>��   As>�P   AsA9P   B�6�AsA^�   AsA9P   AsC�P   B��AsC��   AsC�P   AsFP   B��AsF@�   AsFP   AsH�P   B��AsH��   AsH�P   AsJ�P   BƟSAsK"�   AsJ�P   AsMnP   B�6�AsM��   AsMnP   AsO�P   B�~�AsP�   AsO�P   AsRPP   B��AsRu�   AsRPP   AsT�P   B�F�AsT��   AsT�P   AsW2P   B�HAsWW�   AsW2P   AsY�P   B��AsY��   AsY�P   As\P   B���As\9�   As\P   As^�P   BŽ�As^��   As^�P   As`�P   B��Asa�   As`�P   AscgP   B��QAsc��   AscgP   Ase�P   BŃdAse��   Ase�P   AshIP   B�8Ashn�   AshIP   Asj�P   B��HAsj��   Asj�P   Asm+P   B�O�AsmP�   Asm+P   Aso�P   B��CAso��   Aso�P   AsrP   B��Asr2�   AsrP   Ast~P   BœrAst��   Ast~P   Asv�P   B�N�Asw�   Asv�P   Asy`P   B�-�Asy��   Asy`P   As{�P   B��As{��   As{�P   As~BP   B�~qAs~g�   As~BP   As��P   BŔAs���   As��P   As�$P   B�$vAs�I�   As�$P   As��P   B��As���   As��P   As�P   B��As�+�   As�P   As�wP   B�@ZAs���   As�wP   As��P   BŽAs��   As��P   As�YP   B�>�As�~�   As�YP   As��P   B�(As���   As��P   As�;P   B�R�As�`�   As�;P   As��P   B��_As���   As��P   As�P   B�BcAs�B�   As�P   As��P   B��As���   As��P   As��P   B��CAs�$�   As��P   As�pP   B�%As���   As�pP   As��P   B��As��   As��P   As�RP   B�<NAs�w�   As�RP   As��P   B��As���   As��P   As�4P   B�a�As�Y�   As�4P   As��P   Bĭ�As���   As��P   As�P   BŽ�As�;�   As�P   As��P   B�95As���   As��P   As��P   B��As��   As��P   As�iP   B�t�As���   As�iP   As��P   B�wYAs���   As��P   As�KP   Bı�As�p�   As�KP   As��P   B��3As���   As��P   As�-P   B��KAs�R�   As�-P   AsP   B�GAs���   AsP   As�P   B��-As�4�   As�P   AsǀP   Bć�Asǥ�   AsǀP   As��P   B��=As��   As��P   As�bP   BĀ�Aṡ�   As�bP   As��P   B�As���   As��P   As�DP   B��\As�i�   As�DP   AsӵP   Bċ�As���   AsӵP   As�&P   B�k�As�K�   As�&P   AsؗP   B���Asؼ�   AsؗP   As�P   B�XAs�-�   As�P   As�yP   B�@�Asݞ�   As�yP   As��P   Bį�As��   As��P   As�[P   B�F	As��   As�[P   As��P   B��-As���   As��P   As�=P   B�zAs�b�   As�=P   As�P   B��oAs���   As�P   As�P   B�Z As�D�   As�P   As�P   Bķ�As��   As�P   As�P   B�k�As�&�   As�P   As�rP   B���As��   As�rP   As��P   B�$As��   As��P   As�TP   BÝMAs�y�   As�TP   As��P   B���As���   As��P   As�6P   B���As�[�   As�6P   As��P   BÈ�As���   As��P   AtP   B�@�At=�   AtP   At�P   B���At��   At�P   At�P   B�V�At�   At�P   At	kP   B��