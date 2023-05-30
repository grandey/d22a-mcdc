CDF   �   
      time       bnds      lon       lat          6   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-09-15T06:44:03Z   description       CMIP6 historical   title         =CNRM-ESM2-1 model output prepared for CMIP6 / CMIP historical      activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.13   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Vhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.historical.none.r1i1p1f2     grid      ldata regridded to a T127 gaussian grid (128x256 latlon) from a native atmosphere T127l reduced gaussian grid   
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         atmos      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rsdt   variant_label         r1i1p1f2   EXPID         "CNRM-ESM2-1_historical_r1i1p1f2_v2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          92ddb3d0d8ce79f498d792fc8e559dcf   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      history      FTue May 30 16:58:19 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CNRM-ESM2-1_r1i1p1f2/rsdt_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.ScenarioMIP.CNRM-CERFACS.CNRM-ESM2-1.ssp245.r1i1p1f2.Amon.rsdt.gr.v20190328/rsdt_Amon_CNRM-ESM2-1_ssp245_r1i1p1f2_gr_201501-210012.yearmean.mul.areacella_piControl_v20181115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CNRM-ESM2-1_r1i1p1f2/rsdt_CNRM-ESM2-1_r1i1p1f2_ssp245.mergetime.nc
Tue May 30 16:58:19 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.yearmean.mul.areacella_piControl_v20181115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CNRM-ESM2-1_r1i1p1f2/rsdt_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc
Fri Nov 04 04:42:10 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.fx.areacella.gr.v20181115/areacella_fx_CNRM-ESM2-1_piControl_r1i1p1f2_gr.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.yearmean.mul.areacella_piControl_v20181115.fldsum.nc
none     tracking_id       1hdl:21.14100/64c42fce-8f07-458f-9dbd-c122da3eae13      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                      standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       at the top of the atmosphere   positive      down   history       none   cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   \@�Aq�6�   Aq�P   Aq��P   \?hAq���   Aq��P   Aq��P   \@1Aq��   Aq��P   Aq�dP   \<�Aq���   Aq�dP   Aq��P   \9�Aq���   Aq��P   Aq�FP   \7�Aq�k�   Aq�FP   Aq��P   \8>Aq���   Aq��P   Aq�(P   \8�Aq�M�   Aq�(P   Aq��P   \<�Aq���   Aq��P   Aq�
P   \?�Aq�/�   Aq�
P   Aq�{P   \CAq���   Aq�{P   Aq��P   \A�Aq��   Aq��P   Aq�]P   \>�AqĂ�   Aq�]P   Aq��P   \=MAq���   Aq��P   Aq�?P   \<�Aq�d�   Aq�?P   Aq˰P   \:YAq���   Aq˰P   Aq�!P   \9�Aq�F�   Aq�!P   AqВP   \8�Aqз�   AqВP   Aq�P   \;�Aq�(�   Aq�P   Aq�tP   \>�Aqՙ�   Aq�tP   Aq��P   \?FAq�
�   Aq��P   Aq�VP   \A�Aq�{�   Aq�VP   Aq��P   \AVAq���   Aq��P   Aq�8P   \>;Aq�]�   Aq�8P   Aq�P   \<]Aq���   Aq�P   Aq�P   \8�Aq�?�   Aq�P   Aq�P   \8"Aq��   Aq�P   Aq��P   \6!Aq�!�   Aq��P   Aq�mP   \4�Aq��   Aq�mP   Aq��P   \5#Aq��   Aq��P   Aq�OP   \96Aq�t�   Aq�OP   Aq��P   \<�Aq���   Aq��P   Aq�1P   \;�Aq�V�   Aq�1P   Aq��P   \;Aq���   Aq��P   Aq�P   \AHAq�8�   Aq�P   Aq��P   \<�Aq���   Aq��P   Aq��P   \9]Aq��   Aq��P   ArfP   \7Ar��   ArfP   Ar�P   \7Ar��   Ar�P   ArHP   \5�Arm�   ArHP   Ar�P   \6�Ar��   Ar�P   Ar*P   \<^ArO�   Ar*P   Ar�P   \BAr��   Ar�P   ArP   \D�Ar1�   ArP   Ar}P   \I\Ar��   Ar}P   Ar�P   \E%Ar�   Ar�P   Ar_P   \@*Ar��   Ar_P   Ar�P   \;�Ar��   Ar�P   ArAP   \9�Arf�   ArAP   Ar�P   \7�Ar��   Ar�P   Ar!#P   \8Ar!H�   Ar!#P   Ar#�P   \4hAr#��   Ar#�P   Ar&P   \4�Ar&*�   Ar&P   Ar(vP   \9Ar(��   Ar(vP   Ar*�P   \A�Ar+�   Ar*�P   Ar-XP   \==Ar-}�   Ar-XP   Ar/�P   \B�Ar/��   Ar/�P   Ar2:P   \?�Ar2_�   Ar2:P   Ar4�P   \B�Ar4��   Ar4�P   Ar7P   \=�Ar7A�   Ar7P   Ar9�P   \;Ar9��   Ar9�P   Ar;�P   \6�Ar<#�   Ar;�P   Ar>oP   \6KAr>��   Ar>oP   Ar@�P   \5�ArA�   Ar@�P   ArCQP   \8�ArCv�   ArCQP   ArE�P   \A�ArE��   ArE�P   ArH3P   \JArHX�   ArH3P   ArJ�P   \M�ArJ��   ArJ�P   ArMP   \N�ArM:�   ArMP   ArO�P   \GWArO��   ArO�P   ArQ�P   \AMArR�   ArQ�P   ArThP   \;�ArT��   ArThP   ArV�P   \8ArV��   ArV�P   ArYJP   \7ArYo�   ArYJP   Ar[�P   \9�Ar[��   Ar[�P   Ar^,P   \=�Ar^Q�   Ar^,P   Ar`�P   \DAr`��   Ar`�P   ArcP   \I}Arc3�   ArcP   AreP   \F�Are��   AreP   Arg�P   \D#Arh�   Arg�P   ArjaP   \D:Arj��   ArjaP   Arl�P   \?Arl��   Arl�P   AroCP   \<OAroh�   AroCP   Arq�P   \9�Arq��   Arq�P   Art%P   \;�ArtJ�   Art%P   Arv�P   \A�Arv��   Arv�P   AryP   \O�Ary,�   AryP   Ar{xP   \OpAr{��   Ar{xP   Ar}�P   \MAr~�   Ar}�P   Ar�ZP   \LqAr��   Ar�ZP   Ar��P   \H�Ar���   Ar��P   Ar�<P   \DdAr�a�   Ar�<P   Ar��P   \@.Ar���   Ar��P   Ar�P   \;�Ar�C�   Ar�P   Ar��P   \>Ar���   Ar��P   Ar� P   \E,Ar�%�   Ar� P   Ar�qP   \H�Ar���   Ar�qP   Ar��P   \ScAr��   Ar��P   Ar�SP   \W�Ar�x�   Ar�SP   Ar��P   \S`Ar���   Ar��P   Ar�5P   \K�Ar�Z�   Ar�5P   Ar��P   \A�Ar���   Ar��P   Ar�P   \BFAr�<�   Ar�P   Ar��P   \>[Ar���   Ar��P   Ar��P   \>gAr��   Ar��P   Ar�jP   \D�Ar���   Ar�jP   Ar��P   \S Ar� �   Ar��P   Ar�LP   \^�Ar�q�   Ar�LP   Ar��P   \`�Ar���   Ar��P   Ar�.P   \Y�Ar�S�   Ar�.P   Ar��P   \UvAr���   Ar��P   Ar�P   \I�Ar�5�   Ar�P   Ar��P   \A�Ar���   Ar��P   Ar��P   \?kAr��   Ar��P   Ar�cP   \?Ar���   Ar�cP   Ar��P   \>�Ar���   Ar��P   Ar�EP   \D�Ar�j�   Ar�EP   ArĶP   \K�Ar���   ArĶP   Ar�'P   \P�Ar�L�   Ar�'P   ArɘP   \PRArɽ�   ArɘP   Ar�	P   \P�Ar�.�   Ar�	P   Ar�zP   \JVArΟ�   Ar�zP   Ar��P   \J�Ar��   Ar��P   Ar�\P   \CQArӁ�   Ar�\P   Ar��P   \@"Ar���   Ar��P   Ar�>P   \@9Ar�c�   Ar�>P   ArگP   \@Ar���   ArگP   Ar� P   \E�Ar�E�   Ar� P   ArߑP   \PdAr߶�   ArߑP   Ar�P   \Y@Ar�'�   Ar�P   Ar�sP   \XbAr��   Ar�sP   Ar��P   \XAr�	�   Ar��P   Ar�UP   \N%Ar�z�   Ar�UP   Ar��P   \M�Ar���   Ar��P   Ar�7P   \C-Ar�\�   Ar�7P   Ar�P   \>XAr���   Ar�P   Ar�P   \>�Ar�>�   Ar�P   Ar��P   \AQAr���   Ar��P   Ar��P   \LAr� �   Ar��P   Ar�lP   \Y|Ar���   Ar�lP   Ar��P   \WAr��   Ar��P   Ar�NP   \V�Ar�s�   Ar�NP   As�P   \R�As��   As�P   As0P   \F�AsU�   As0P   As�P   \@As��   As�P   As	P   \>�As	7�   As	P   As�P   \<�As��   As�P   As�P   \?�As�   As�P   AseP   \JYAs��   AseP   As�P   \R8As��   As�P   AsGP   \X�Asl�   AsGP   As�P   \U�As��   As�P   As)P   \V�AsN�   As)P   As�P   \H�As��   As�P   AsP   \C�As0�   AsP   As!|P   \<�As!��   As!|P   As#�P   \;]As$�   As#�P   As&^P   \8�As&��   As&^P   As(�P   \9As(��   As(�P   As+@P   \8"As+e�   As+@P   As-�P   \=6As-��   As-�P   As0"P   \D�As0G�   As0"P   As2�P   \I7As2��   As2�P   As5P   \JAs5)�   As5P   As7uP   \LPAs7��   As7uP   As9�P   \E�As:�   As9�P   As<WP   \>0As<|�   As<WP   As>�P   \:�As>��   As>�P   AsA9P   \:AsA^�   AsA9P   AsC�P   \9�AsC��   AsC�P   AsFP   \<
AsF@�   AsFP   AsH�P   \CUAsH��   AsH�P   AsJ�P   \L]AsK"�   AsJ�P   AsMnP   \Q�AsM��   AsMnP   AsO�P   \T�AsP�   AsO�P   AsRPP   \N�AsRu�   AsRPP   AsT�P   \H�AsT��   AsT�P   AsW2P   \C+AsWW�   AsW2P   AsY�P   \A�AsY��   AsY�P   As\P   \<�As\9�   As\P   As^�P   \:�As^��   As^�P   As`�P   \8�Asa�   As`�P   AscgP   \:BAsc��   AscgP   Ase�P   \>=Ase��   Ase�P   AshIP   \F�Ashn�   AshIP   Asj�P   \G�Asj��   Asj�P   Asm+P   \L�AsmP�   Asm+P   Aso�P   \G�Aso��   Aso�P   AsrP   \G�Asr2�   AsrP   Ast~P   \BNAst��   Ast~P   Asv�P   \=�Asw�   Asv�P   Asy`P   \8cAsy��   Asy`P   As{�P   \7$As{��   As{�P   As~BP   \7�As~g�   As~BP   As��P   \;�As���   As��P   As�$P   \B�As�I�   As�$P   As��P   \HWAs���   As��P   As�P   \N�As�+�   As�P   As�wP   \P/As���   As�wP   As��P   \JAs��   As��P   As�YP   \BSAs�~�   As�YP   As��P   \>yAs���   As��P   As�;P   \;�As�`�   As�;P   As��P   \8LAs���   As��P   As�P   \9LAs�B�   As�P   As��P   \>As���   As��P   As��P   \D�As�$�   As��P   As�pP   \DAs���   As�pP   As��P   \EAs��   As��P   As�RP   \E�As�w�   As�RP   As��P   \A�As���   As��P   As�4P   \<�As�Y�   As�4P   As��P   \9XAs���   As��P   As�P   \8-As�;�   As�P   As��P   \8�As���   As��P   As��P   \9&As��   As��P   As�iP   \?�As���   As�iP   As��P   \EnAs���   As��P   As�KP   \I�As�p�   As�KP   As��P   \GXAs���   As��P   As�-P   \C�As�R�   As�-P   AsP   \>�As���   AsP   As�P   \=As�4�   As�P   AsǀP   \:VAsǥ�   AsǀP   As��P   \7�As��   As��P   As�bP   \6�Aṡ�   As�bP   As��P   \6hAs���   As��P   As�DP   \6�As�i�   As�DP   AsӵP   \;As���   AsӵP   As�&P   \?pAs�K�   As�&P   AsؗP   \CsAsؼ�   AsؗP   As�P   \B�As�-�   As�P   As�yP   \B�Asݞ�   As�yP   As��P   \?kAs��   As��P   As�[P   \>�As��   As�[P   As��P   \8�As���   As��P   As�=P   \67As�b�   As�=P   As�P   \6As���   As�P   As�P   \7NAs�D�   As�P   As�P   \:#As��   As�P   As�P   \@uAs�&�   As�P   As�rP   \EAs��   As�rP   As��P   \L
As��   As��P   As�TP   \H�As�y�   As�TP   As��P   \C�As���   As��P   As�6P   \?�As�[�   As�6P   As��P   \;�As���   As��P   AtP   \9?At=�   AtP   At�P   \7�At��   At�P   At�P   \9�At�   At�P   At	kP   \@}