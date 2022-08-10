CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �GISS-E2.1G (2019): 
aerosol: Varies with physics-version (p==1 none, p==3 OMA, p==4 TOMAS, p==5 MATRIX)
atmos: GISS-E2.1 (2.5x2 degree; 144 x 90 longitude/latitude; 40 levels; top level 0.1 hPa)
atmosChem: Varies with physics-version (p==1 Non-interactive, p>1 GPUCCINI)
land: GISS LSM
landIce: none
ocean: GISS Ocean (GO1, 1 degree; 360 x 180 longitude/latitude; 40 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: GISS SI   institution       <Goddard Institute for Space Studies, New York, NY 10025, USA   activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    contact        Kenneth Lo (cdkkl@giss.nasa.gov)   creation_date         2020-03-19T16:43:49Z   data_specs_version        01.00.23   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     forcing_index               	frequency         year   further_info_url      Shttps://furtherinfo.es-doc.org/CMIP6.NASA-GISS.GISS-E2-1-G.historical.none.r1i1p5f1    grid      -atmospheric grid: 144x90, ocean grid: 288x180      
grid_label        gn     history      �Wed Aug 10 15:22:22 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/GISS-E2-1-G_r1i1p5f1/zostoga_GISS-E2-1-G_r1i1p5f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/GISS-E2-1-G_r1i1p5f1/CMIP6.ScenarioMIP.NASA-GISS.GISS-E2-1-G.ssp585.r1i1p5f1.Omon.zostoga.gn.v20200115/zostoga_Omon_GISS-E2-1-G_ssp585_r1i1p5f1_gn_201501-210012.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/GISS-E2-1-G_r1i1p5f1/zostoga_GISS-E2-1-G_r1i1p5f1_ssp585.mergetime.nc
Wed Aug 10 15:22:21 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/GISS-E2-1-G_r1i1p5f1/CMIP6.CMIP.NASA-GISS.GISS-E2-1-G.historical.r1i1p5f1.Omon.zostoga.gn.v20190905/zostoga_Omon_GISS-E2-1-G_historical_r1i1p5f1_gn_185001-201412.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/GISS-E2-1-G_r1i1p5f1/zostoga_GISS-E2-1-G_r1i1p5f1_historical.mergetime.nc
Thu Apr 07 23:32:37 2022: cdo -O -s -selname,zostoga -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/GISS-E2-1-G_r1i1p5f1/CMIP6.CMIP.NASA-GISS.GISS-E2-1-G.historical.r1i1p5f1.Omon.zostoga.gn.v20190905/zostoga_Omon_GISS-E2-1-G_historical_r1i1p5f1_gn_185001-201412.1d.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/GISS-E2-1-G_r1i1p5f1/CMIP6.CMIP.NASA-GISS.GISS-E2-1-G.historical.r1i1p5f1.Omon.zostoga.gn.v20190905/zostoga_Omon_GISS-E2-1-G_historical_r1i1p5f1_gn_185001-201412.1d.yearmean.fldmean.nc
Thu Apr 07 23:32:36 2022: cdo -O -s --reduce_dim -selname,zostoga -yearmean /Users/benjamin/Data/p22b/CMIP6/zostoga/GISS-E2-1-G_r1i1p5f1/CMIP6.CMIP.NASA-GISS.GISS-E2-1-G.historical.r1i1p5f1.Omon.zostoga.gn.v20190905/zostoga_Omon_GISS-E2-1-G_historical_r1i1p5f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/GISS-E2-1-G_r1i1p5f1/CMIP6.CMIP.NASA-GISS.GISS-E2-1-G.historical.r1i1p5f1.Omon.zostoga.gn.v20190905/zostoga_Omon_GISS-E2-1-G_historical_r1i1p5f1_gn_185001-201412.1d.yearmean.nc
2020-03-19T16:43:49Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.   initialization_index            institution_id        	NASA-GISS      mip_era       CMIP6      model_id      E214Tmatrixf10aF40oQ40     nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p5     parent_mip_era        CMIP6      parent_source_id      GISS-E2-1-G    parent_time_units         days since 2000-1-1    parent_variant_label      r1i1p5f1   physics_index               product       model-output   realization_index               realm         ocean      
references        'https://data.giss.nasa.gov/modelE/cmip6    	source_id         GISS-E2-1-G    source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(21 March 2018) MD5:d211846f9d8f448e2718291a152c71ca     title         %GISS-E2-1-G output prepared for CMIP6      tracking_id       1hdl:21.14100/22752263-ca60-48fe-9025-d72ba3a77bf1      variable_id       zostoga    variant_label         r1i1p5f1   license      cCMIP6 model data produced by NASA Goddard Institute for Space Studies is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.3.2      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               D   	time_bnds                                 L   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               4   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               <   zostoga                       standard_name         ,global_average_thermosteric_sea_level_change   	long_name         ,Global Average Thermosteric Sea Level Change   units         m      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       /There is no CMIP6 request for zosga nor zossga.    history       o2020-03-19T16:43:49Z altered by CMOR: replaced missing value flag (-1e+30) with standard missing value (1e+20).             \                Aq���   Aq��P   Aq�P   >���Aq�6�   Aq�P   Aq��P   >���Aq���   Aq��P   Aq��P   >��Aq��   Aq��P   Aq�dP   >�ߑAq���   Aq�dP   Aq��P   >��}Aq���   Aq��P   Aq�FP   >�0XAq�k�   Aq�FP   Aq��P   >�;:Aq���   Aq��P   Aq�(P   >�i$Aq�M�   Aq�(P   Aq��P   >��~Aq���   Aq��P   Aq�
P   >틀Aq�/�   Aq�
P   Aq�{P   >���Aq���   Aq�{P   Aq��P   >�YAq��   Aq��P   Aq�]P   >�gyAqĂ�   Aq�]P   Aq��P   >�ĠAq���   Aq��P   Aq�?P   >���Aq�d�   Aq�?P   Aq˰P   >�ۣAq���   Aq˰P   Aq�!P   >�9Aq�F�   Aq�!P   AqВP   >��CAqз�   AqВP   Aq�P   >�Aq�(�   Aq�P   Aq�tP   >�g�Aqՙ�   Aq�tP   Aq��P   >��Aq�
�   Aq��P   Aq�VP   >��Aq�{�   Aq�VP   Aq��P   >�U^Aq���   Aq��P   Aq�8P   >��0Aq�]�   Aq�8P   Aq�P   >�iAq���   Aq�P   Aq�P   >�`yAq�?�   Aq�P   Aq�P   >��Aq��   Aq�P   Aq��P   >�;Aq�!�   Aq��P   Aq�mP   >�~�Aq��   Aq�mP   Aq��P   >�@�Aq��   Aq��P   Aq�OP   >�ۨAq�t�   Aq�OP   Aq��P   >�=�Aq���   Aq��P   Aq�1P   >���Aq�V�   Aq�1P   Aq��P   >�vAq���   Aq��P   Aq�P   >�|dAq�8�   Aq�P   Aq��P   >��&Aq���   Aq��P   Aq��P   >��`Aq��   Aq��P   ArfP   >���Ar��   ArfP   Ar�P   >�˟Ar��   Ar�P   ArHP   >�K�Arm�   ArHP   Ar�P   >���Ar��   Ar�P   Ar*P   >��jArO�   Ar*P   Ar�P   >��Ar��   Ar�P   ArP   >�V�Ar1�   ArP   Ar}P   >�S�Ar��   Ar}P   Ar�P   >�?�Ar�   Ar�P   Ar_P   >��_Ar��   Ar_P   Ar�P   >��Ar��   Ar�P   ArAP   >���Arf�   ArAP   Ar�P   >�WEAr��   Ar�P   Ar!#P   >���Ar!H�   Ar!#P   Ar#�P   ? qAr#��   Ar#�P   Ar&P   ? XCAr&*�   Ar&P   Ar(vP   >���Ar(��   Ar(vP   Ar*�P   ? .KAr+�   Ar*�P   Ar-XP   ? ;Ar-}�   Ar-XP   Ar/�P   >��Ar/��   Ar/�P   Ar2:P   ? CAr2_�   Ar2:P   Ar4�P   ? �KAr4��   Ar4�P   Ar7P   ? ǆAr7A�   Ar7P   Ar9�P   ? @�Ar9��   Ar9�P   Ar;�P   ? U�Ar<#�   Ar;�P   Ar>oP   ?��Ar>��   Ar>oP   Ar@�P   ?jArA�   Ar@�P   ArCQP   ?�ArCv�   ArCQP   ArE�P   ?��ArE��   ArE�P   ArH3P   ?gjArHX�   ArH3P   ArJ�P   ?��ArJ��   ArJ�P   ArMP   ?L�ArM:�   ArMP   ArO�P   ?��ArO��   ArO�P   ArQ�P   ?V	ArR�   ArQ�P   ArThP   ?/�ArT��   ArThP   ArV�P   ?DvArV��   ArV�P   ArYJP   ?^ArYo�   ArYJP   Ar[�P   ?��Ar[��   Ar[�P   Ar^,P   ?��Ar^Q�   Ar^,P   Ar`�P   ?�@Ar`��   Ar`�P   ArcP   ?ɈArc3�   ArcP   AreP   ?^
Are��   AreP   Arg�P   ?G�Arh�   Arg�P   ArjaP   ?0�Arj��   ArjaP   Arl�P   ?,_Arl��   Arl�P   AroCP   ?��Aroh�   AroCP   Arq�P   ?%Arq��   Arq�P   Art%P   ?q�ArtJ�   Art%P   Arv�P   ?)rArv��   Arv�P   AryP   ?9rAry,�   AryP   Ar{xP   ?�1Ar{��   Ar{xP   Ar}�P   ?�@Ar~�   Ar}�P   Ar�ZP   ?p2Ar��   Ar�ZP   Ar��P   ?��Ar���   Ar��P   Ar�<P   ?n=Ar�a�   Ar�<P   Ar��P   ?��Ar���   Ar��P   Ar�P   ?سAr�C�   Ar�P   Ar��P   ?	��Ar���   Ar��P   Ar� P   ?	��Ar�%�   Ar� P   Ar�qP   ?	ؒAr���   Ar�qP   Ar��P   ?
|�Ar��   Ar��P   Ar�SP   ?
Ar�x�   Ar�SP   Ar��P   ?
��Ar���   Ar��P   Ar�5P   ?
zAr�Z�   Ar�5P   Ar��P   ?b<Ar���   Ar��P   Ar�P   ?�Ar�<�   Ar�P   Ar��P   ?&�Ar���   Ar��P   Ar��P   ? Ar��   Ar��P   Ar�jP   ?��Ar���   Ar�jP   Ar��P   ?A�Ar� �   Ar��P   Ar�LP   ?�Ar�q�   Ar�LP   Ar��P   ?;�Ar���   Ar��P   Ar�.P   ?��Ar�S�   Ar�.P   Ar��P   ?mAr���   Ar��P   Ar�P   ?H5Ar�5�   Ar�P   Ar��P   ?LwAr���   Ar��P   Ar��P   ?�`Ar��   Ar��P   Ar�cP   ?F1Ar���   Ar�cP   Ar��P   ?��Ar���   Ar��P   Ar�EP   ?��Ar�j�   Ar�EP   ArĶP   ?#PAr���   ArĶP   Ar�'P   ?\*Ar�L�   Ar�'P   ArɘP   ?Q�Arɽ�   ArɘP   Ar�	P   ?0Ar�.�   Ar�	P   Ar�zP   ?�LArΟ�   Ar�zP   Ar��P   ?�fAr��   Ar��P   Ar�\P   ?�ArӁ�   Ar�\P   Ar��P   ?v�Ar���   Ar��P   Ar�>P   ? jAr�c�   Ar�>P   ArگP   ?��Ar���   ArگP   Ar� P   ?��Ar�E�   Ar� P   ArߑP   ?��Ar߶�   ArߑP   Ar�P   ?�@Ar�'�   Ar�P   Ar�sP   ?v�Ar��   Ar�sP   Ar��P   ?�#Ar�	�   Ar��P   Ar�UP   ?��Ar�z�   Ar�UP   Ar��P   ?�Ar���   Ar��P   Ar�7P   ?z�Ar�\�   Ar�7P   Ar�P   ?�uAr���   Ar�P   Ar�P   ?��Ar�>�   Ar�P   Ar��P   ?I�Ar���   Ar��P   Ar��P   ?>�Ar� �   Ar��P   Ar�lP   ?�Ar���   Ar�lP   Ar��P   ?AAr��   Ar��P   Ar�NP   ?~Ar�s�   Ar�NP   As�P   ?��As��   As�P   As0P   ?�ZAsU�   As0P   As�P   ?��As��   As�P   As	P   ?�>As	7�   As	P   As�P   ?�RAs��   As�P   As�P   ?x�As�   As�P   AseP   ?��As��   AseP   As�P   ?�	As��   As�P   AsGP   ?� Asl�   AsGP   As�P   ?��As��   As�P   As)P   ?w�AsN�   As)P   As�P   ?�As��   As�P   AsP   ?��As0�   AsP   As!|P   ?��As!��   As!|P   As#�P   ?�6As$�   As#�P   As&^P   ?j�As&��   As&^P   As(�P   ?I�As(��   As(�P   As+@P   ?��As+e�   As+@P   As-�P   ?�As-��   As-�P   As0"P   ?9yAs0G�   As0"P   As2�P   ?�JAs2��   As2�P   As5P   ?�*As5)�   As5P   As7uP   ?�As7��   As7uP   As9�P   ?!�As:�   As9�P   As<WP   ?"ihAs<|�   As<WP   As>�P   ?"�pAs>��   As>�P   AsA9P   ?#�AsA^�   AsA9P   AsC�P   ?$[�AsC��   AsC�P   AsFP   ?$u�AsF@�   AsFP   AsH�P   ?$&AsH��   AsH�P   AsJ�P   ?%�&AsK"�   AsJ�P   AsMnP   ?%�AsM��   AsMnP   AsO�P   ?&·AsP�   AsO�P   AsRPP   ?&��AsRu�   AsRPP   AsT�P   ?'HAsT��   AsT�P   AsW2P   ?'�cAsWW�   AsW2P   AsY�P   ?))�AsY��   AsY�P   As\P   ?*��As\9�   As\P   As^�P   ?*�lAs^��   As^�P   As`�P   ?*=0Asa�   As`�P   AscgP   ?+�_Asc��   AscgP   Ase�P   ?,;Ase��   Ase�P   AshIP   ?-��Ashn�   AshIP   Asj�P   ?.FJAsj��   Asj�P   Asm+P   ?.}�AsmP�   Asm+P   Aso�P   ?/K�Aso��   Aso�P   AsrP   ?/�WAsr2�   AsrP   Ast~P   ?0�#Ast��   Ast~P   Asv�P   ?0�Asw�   Asv�P   Asy`P   ?1�IAsy��   Asy`P   As{�P   ?2�;As{��   As{�P   As~BP   ?3�bAs~g�   As~BP   As��P   ?59�As���   As��P   As�$P   ?60�As�I�   As�$P   As��P   ?6��As���   As��P   As�P   ?7�As�+�   As�P   As�wP   ?8D&As���   As�wP   As��P   ?8ƕAs��   As��P   As�YP   ?9x�As�~�   As�YP   As��P   ?:'{As���   As��P   As�;P   ?;vAAs�`�   As�;P   As��P   ?=}gAs���   As��P   As�P   ?=�$As�B�   As�P   As��P   ?>�[As���   As��P   As��P   ?@/As�$�   As��P   As�pP   ?AZ�As���   As�pP   As��P   ?B0�As��   As��P   As�RP   ?B��As�w�   As�RP   As��P   ?D`XAs���   As��P   As�4P   ?E��As�Y�   As�4P   As��P   ?F}�As���   As��P   As�P   ?F�As�;�   As�P   As��P   ?H��As���   As��P   As��P   ?J%As��   As��P   As�iP   ?JsPAs���   As�iP   As��P   ?KJ�As���   As��P   As�KP   ?LcAs�p�   As�KP   As��P   ?MkfAs���   As��P   As�-P   ?OLAs�R�   As�-P   AsP   ?O��As���   AsP   As�P   ?P�0As�4�   As�P   AsǀP   ?R5Asǥ�   AsǀP   As��P   ?R�As��   As��P   As�bP   ?SϘAṡ�   As�bP   As��P   ?US�As���   As��P   As�DP   ?W��As�i�   As�DP   AsӵP   ?X��As���   AsӵP   As�&P   ?X�As�K�   As�&P   AsؗP   ?Z[�Asؼ�   AsؗP   As�P   ?[�lAs�-�   As�P   As�yP   ?]ϪAsݞ�   As�yP   As��P   ?^[#As��   As��P   As�[P   ?_��As��   As�[P   As��P   ?`��As���   As��P   As�=P   ?bAs�b�   As�=P   As�P   ?d7AAs���   As�P   As�P   ?eAs�D�   As�P   As�P   ?e��As��   As�P   As�P   ?g}As�&�   As�P   As�rP   ?h|GAs��   As�rP   As��P   ?i�-As��   As��P   As�TP   ?j�6As�y�   As�TP   As��P   ?l-�As���   As��P   As�6P   ?m�As�[�   As�6P   As��P   ?n�'As���   As��P   AtP   ?popAt=�   AtP   At�P   ?q֖At��   At�P   At�P   ?r��At�   At�P   At	kP   ?s�