CDF   �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       0MIROC6 (2017): 
aerosol: SPRINTARS6.0
atmos: CCSR AGCM (T85; 256 x 128 longitude/latitude; 81 levels; top level 0.004 hPa)
atmosChem: none
land: MATSIRO6.0
landIce: none
ocean: COCO4.9 (tripolar primarily 1deg; 360 x 256 longitude/latitude; 63 levels; top grid cell 0-2 m)
ocnBgchem: none
seaIce: COCO4.9   institution      QJAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-01-30T03:49:43Z   data_specs_version        01.00.28   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Jhttps://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.historical.none.r1i1p1f1     grid      #native atmosphere T85 Gaussian grid    
grid_label        gn     history      Wed Nov 09 18:59:29 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/MIROC6_r1i1p1f1/rsdt_MIROC6_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/MIROC6_r1i1p1f1/CMIP6.ScenarioMIP.MIROC.MIROC6.ssp585.r1i1p1f1.Amon.rsdt.gn.v20190627/rsdt_Amon_MIROC6_ssp585_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp585_v20190627.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/MIROC6_r1i1p1f1/rsdt_MIROC6_r1i1p1f1_ssp585.mergetime.nc
Wed Nov 09 18:59:27 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsdt.gn.v20190311/rsdt_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.yearmean.mul.areacella_historical_v20190311.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsdt.gn.v20190311/rsdt_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.yearmean.mul.areacella_historical_v20190311.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/MIROC6_r1i1p1f1/rsdt_MIROC6_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 05:38:06 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsdt.gn.v20190311/rsdt_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.fx.areacella.gn.v20190311/areacella_fx_MIROC6_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsdt.gn.v20190311/rsdt_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.yearmean.mul.areacella_historical_v20190311.fldsum.nc
2019-01-30T03:49:43Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.   initialization_index            institution_id        MIROC      mip_era       CMIP6      nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      MIROC6     parent_time_units         days since 3200-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         MIROC6     source_type       	AOGCM AER      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(06 November 2018) MD5:0728c79344e0f262bb76e4f9ff0d9afc      title          MIROC6 output prepared for CMIP6   variable_id       rsdt   variant_label         r1i1p1f1   license      !CMIP6 model data produced by MIROC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      cmor_version      3.3.2      tracking_id       1hdl:21.14100/08456e26-5450-4895-a7c2-b746ca322beb      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                      standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      original_name         OSRD   original_units        W/m**2     history       �2019-01-30T03:49:43Z altered by CMOR: Converted units from 'W/m**2' to 'W m-2'. 2019-01-30T03:49:43Z altered by CMOR: replaced missing value flag (-999) with standard missing value (1e+20). 2019-01-30T03:49:43Z altered by CMOR: Inverted axis: lat.    cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   \�Aq�6�   Aq�P   Aq��P   \�Aq���   Aq��P   Aq��P   \�Aq��   Aq��P   Aq�dP   \9Aq���   Aq�dP   Aq��P   \Aq���   Aq��P   Aq�FP   \&Aq�k�   Aq�FP   Aq��P   \�Aq���   Aq��P   Aq�(P   \�Aq�M�   Aq�(P   Aq��P   \�Aq���   Aq��P   Aq�
P   \�Aq�/�   Aq�
P   Aq�{P   \�Aq���   Aq�{P   Aq��P   \�Aq��   Aq��P   Aq�]P   \AqĂ�   Aq�]P   Aq��P   \wAq���   Aq��P   Aq�?P   \7Aq�d�   Aq�?P   Aq˰P   \�Aq���   Aq˰P   Aq�!P   \�Aq�F�   Aq�!P   AqВP   \Aqз�   AqВP   Aq�P   \�Aq�(�   Aq�P   Aq�tP   \Aqՙ�   Aq�tP   Aq��P   \WAq�
�   Aq��P   Aq�VP   \�Aq�{�   Aq�VP   Aq��P   \�Aq���   Aq��P   Aq�8P   \kAq�]�   Aq�8P   Aq�P   \�Aq���   Aq�P   Aq�P   \Aq�?�   Aq�P   Aq�P   \�Aq��   Aq�P   Aq��P   \tAq�!�   Aq��P   Aq�mP   \*Aq��   Aq�mP   Aq��P   \iAq��   Aq��P   Aq�OP   \�Aq�t�   Aq�OP   Aq��P   \�Aq���   Aq��P   Aq�1P   \�Aq�V�   Aq�1P   Aq��P   \WAq���   Aq��P   Aq�P   \�Aq�8�   Aq�P   Aq��P   \Aq���   Aq��P   Aq��P   \�Aq��   Aq��P   ArfP   \]Ar��   ArfP   Ar�P   \�Ar��   Ar�P   ArHP   \Arm�   ArHP   Ar�P   \�Ar��   Ar�P   Ar*P   \�ArO�   Ar*P   Ar�P   \�Ar��   Ar�P   ArP   \ Ar1�   ArP   Ar}P   \$PAr��   Ar}P   Ar�P   \ JAr�   Ar�P   Ar_P   \�Ar��   Ar_P   Ar�P   \�Ar��   Ar�P   ArAP   \�Arf�   ArAP   Ar�P   \Ar��   Ar�P   Ar!#P   \�Ar!H�   Ar!#P   Ar#�P   \�Ar#��   Ar#�P   Ar&P   \DAr&*�   Ar&P   Ar(vP   \\Ar(��   Ar(vP   Ar*�P   \*Ar+�   Ar*�P   Ar-XP   \gAr-}�   Ar-XP   Ar/�P   \	Ar/��   Ar/�P   Ar2:P   \�Ar2_�   Ar2:P   Ar4�P   \Ar4��   Ar4�P   Ar7P   \(Ar7A�   Ar7P   Ar9�P   \;Ar9��   Ar9�P   Ar;�P   \9Ar<#�   Ar;�P   Ar>oP   \Ar>��   Ar>oP   Ar@�P   \ArA�   Ar@�P   ArCQP   \ArCv�   ArCQP   ArE�P   \�ArE��   ArE�P   ArH3P   \!�ArHX�   ArH3P   ArJ�P   \(�ArJ��   ArJ�P   ArMP   \)�ArM:�   ArMP   ArO�P   \"+ArO��   ArO�P   ArQ�P   \ArR�   ArQ�P   ArThP   \HArT��   ArThP   ArV�P   \KArV��   ArV�P   ArYJP   \UArYo�   ArYJP   Ar[�P   \|Ar[��   Ar[�P   Ar^,P   \�Ar^Q�   Ar^,P   Ar`�P   \Ar`��   Ar`�P   ArcP   \$kArc3�   ArcP   AreP   \�Are��   AreP   Arg�P   \cArh�   Arg�P   ArjaP   \)Arj��   ArjaP   Arl�P   \HArl��   Arl�P   AroCP   \Aroh�   AroCP   Arq�P   \Arq��   Arq�P   Art%P   \�ArtJ�   Art%P   Arv�P   \�Arv��   Arv�P   AryP   \&�Ary,�   AryP   Ar{xP   \*YAr{��   Ar{xP   Ar}�P   \(3Ar~�   Ar}�P   Ar�ZP   \'CAr��   Ar�ZP   Ar��P   \ �Ar���   Ar��P   Ar�<P   \�Ar�a�   Ar�<P   Ar��P   \XAr���   Ar��P   Ar�P   \,Ar�C�   Ar�P   Ar��P   \�Ar���   Ar��P   Ar� P   \ @Ar�%�   Ar� P   Ar�qP   \$Ar���   Ar�qP   Ar��P   \.=Ar��   Ar��P   Ar�SP   \/9Ar�x�   Ar�SP   Ar��P   \.Ar���   Ar��P   Ar�5P   \&�Ar�Z�   Ar�5P   Ar��P   \Ar���   Ar��P   Ar�P   \�Ar�<�   Ar�P   Ar��P   \�Ar���   Ar��P   Ar��P   \�Ar��   Ar��P   Ar�jP   \�Ar���   Ar�jP   Ar��P   \*�Ar� �   Ar��P   Ar�LP   \9�Ar�q�   Ar�LP   Ar��P   \;'Ar���   Ar��P   Ar�.P   \4dAr�S�   Ar�.P   Ar��P   \,�Ar���   Ar��P   Ar�P   \$�Ar�5�   Ar�P   Ar��P   \&Ar���   Ar��P   Ar��P   \�Ar��   Ar��P   Ar�cP   \�Ar���   Ar�cP   Ar��P   \Ar���   Ar��P   Ar�EP   \�Ar�j�   Ar�EP   ArĶP   \'	Ar���   ArĶP   Ar�'P   \( Ar�L�   Ar�'P   ArɘP   \+9Arɽ�   ArɘP   Ar�	P   \+�Ar�.�   Ar�	P   Ar�zP   \%�ArΟ�   Ar�zP   Ar��P   \"JAr��   Ar��P   Ar�\P   \yArӁ�   Ar�\P   Ar��P   \YAr���   Ar��P   Ar�>P   \mAr�c�   Ar�>P   ArگP   \�Ar���   ArگP   Ar� P   \!1Ar�E�   Ar� P   ArߑP   \+iAr߶�   ArߑP   Ar�P   \3�Ar�'�   Ar�P   Ar�sP   \/�Ar��   Ar�sP   Ar��P   \2�Ar�	�   Ar��P   Ar�UP   \)Ar�z�   Ar�UP   Ar��P   \(�Ar���   Ar��P   Ar�7P   \�Ar�\�   Ar�7P   Ar�P   \�Ar���   Ar�P   Ar�P   \Ar�>�   Ar�P   Ar��P   \�Ar���   Ar��P   Ar��P   \$Ar� �   Ar��P   Ar�lP   \4DAr���   Ar�lP   Ar��P   \1�Ar��   Ar��P   Ar�NP   \1�Ar�s�   Ar�NP   As�P   \*As��   As�P   As0P   \!�AsU�   As0P   As�P   \jAs��   As�P   As	P   \As	7�   As	P   As�P   \�As��   As�P   As�P   \As�   As�P   AseP   \%�As��   AseP   As�P   \-6As��   As�P   AsGP   \0Asl�   AsGP   As�P   \0�As��   As�P   As)P   \1�AsN�   As)P   As�P   \$As��   As�P   AsP   \=As0�   AsP   As!|P   \As!��   As!|P   As#�P   \�As$�   As#�P   As&^P   \As&��   As&^P   As(�P   \�As(��   As(�P   As+@P   \}As+e�   As+@P   As-�P   \sAs-��   As-�P   As0"P   \ As0G�   As0"P   As2�P   \ �As2��   As2�P   As5P   \%As5)�   As5P   As7uP   \'XAs7��   As7uP   As9�P   \ �As:�   As9�P   As<WP   \�As<|�   As<WP   As>�P   \LAs>��   As>�P   AsA9P   \MAsA^�   AsA9P   AsC�P   \6AsC��   AsC�P   AsFP   \�AsF@�   AsFP   AsH�P   \xAsH��   AsH�P   AsJ�P   \'RAsK"�   AsJ�P   AsMnP   \,wAsM��   AsMnP   AsO�P   \+�AsP�   AsO�P   AsRPP   \)�AsRu�   AsRPP   AsT�P   \#�AsT��   AsT�P   AsW2P   \qAsWW�   AsW2P   AsY�P   \JAsY��   AsY�P   As\P   \=As\9�   As\P   As^�P   \(As^��   As^�P   As`�P   \&Asa�   As`�P   AscgP   \Asc��   AscgP   Ase�P   \sAse��   Ase�P   AshIP   \"Ashn�   AshIP   Asj�P   \"�Asj��   Asj�P   Asm+P   \$nAsmP�   Asm+P   Aso�P   \"�Aso��   Aso�P   AsrP   \"�Asr2�   AsrP   Ast~P   \�Ast��   Ast~P   Asv�P   \�Asw�   Asv�P   Asy`P   \�Asy��   Asy`P   As{�P   \vAs{��   As{�P   As~BP   \As~g�   As~BP   As��P   \AAs���   As��P   As�$P   \�As�I�   As�$P   As��P   \#cAs���   As��P   As�P   \)~As�+�   As�P   As�wP   \'�As���   As�wP   As��P   \$�As��   As��P   As�YP   \�As�~�   As�YP   As��P   \�As���   As��P   As�;P   \}As�`�   As�;P   As��P   \�As���   As��P   As�P   \�As�B�   As�P   As��P   \FAs���   As��P   As��P   \GAs�$�   As��P   As�pP   \HAs���   As�pP   As��P   \ 2As��   As��P   As�RP   \ �As�w�   As�RP   As��P   \�As���   As��P   As�4P   \�As�Y�   As�4P   As��P   \�As���   As��P   As�P   \nAs�;�   As�P   As��P   \jAs���   As��P   As��P   \{As��   As��P   As�iP   \�As���   As�iP   As��P   \ uAs���   As��P   As�KP   \! As�p�   As�KP   As��P   \"wAs���   As��P   As�-P   \�As�R�   As�-P   AsP   \�As���   AsP   As�P   \�As�4�   As�P   AsǀP   \�Asǥ�   AsǀP   As��P   \�As��   As��P   As�bP   \�Aṡ�   As�bP   As��P   \0As���   As��P   As�DP   \<As�i�   As�DP   AsӵP   \8As���   AsӵP   As�&P   \�As�K�   As�&P   AsؗP   \Asؼ�   AsؗP   As�P   \�As�-�   As�P   As�yP   \�Asݞ�   As�yP   As��P   \�As��   As��P   As�[P   \ZAs��   As�[P   As��P   \5As���   As��P   As�=P   \�As�b�   As�=P   As�P   \OAs���   As�P   As�P   \As�D�   As�P   As�P   \lAs��   As�P   As�P   \�As�&�   As�P   As�rP   \ As��   As�rP   As��P   \#~As��   As��P   As�TP   \#�As�y�   As�TP   As��P   \�As���   As��P   As�6P   \�As�[�   As�6P   As��P   \|As���   As��P   AtP   \�At=�   AtP   At�P   \At��   At�P   At�P   \BAt�   At�P   At	kP   \